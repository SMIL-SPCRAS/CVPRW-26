import os
import random
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import List, Dict, Optional, Set, Tuple, Any

import numpy as np
import torch
import polars as pl
from tqdm import tqdm

from transformers import Qwen3VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info


@dataclass
class Config:
    video_dir: Path = Path("//SMIL/Databases_3/ABAW2024/data/Chunk/train")
    input_csv: Path = Path("//SMIL/Databases_3/ABAW2024/data/Chunk/train_segment.csv")
    output_csv: Path = Path("train_segment_embeddings_Qwen3-VL-4B-Instruct.csv")

    # HF model id (скачает в cache HuggingFace)
    # model_name: str = "Qwen/Qwen3-VL-4B-Instruct"
    model_name: str = (
        "models--Qwen--Qwen3-VL-4B-Instruct/snapshots/ebb281ec70b05090aa6165b016eac8ec08e71b17"
    )

    seed: int = 42
    max_new_tokens: int = 100
    torch_dtype: torch.dtype = torch.bfloat16

    video_fps: float = 4.0
    max_frames: int = 16

    save_every: int = 10

    # embeddings + pickle saving
    embeddings_dir: Path = Path("traain_segment_embeddings_Qwen3-VL-4B-Instruct")
    emb_dtype: str = "float16"  # "float16" or "float32"

    # Pickle mode:
    #   "single" -> one master pickle with dict[video_name] = payload
    #   "per_video" -> separate pickle per video
    pickle_mode: str = "single"  # "single" or "per_video"
    master_pickle_path: Path = Path("train_segment_embeddings_Qwen3-VL-4B-Instruct.pkl")

    # Visual token isolation: best-effort; if cannot isolate -> fallback to "video + anchor text"
    visual_anchor_text: str = "Describe only the visible content."


PROMPT_TEMPLATE: str = (
    "You are an expert in multimodal affect analysis. Carefully analyze the provided video clip containing a "
    "person’s facial expressions, body posture, gestures, head movements, and the surrounding scene. "
    "Describe the person’s affect in terms of valence (pleasant ↔ unpleasant) and arousal (low ↔ high). "
    "You may mention uncertainty or mixed/ambiguous affect when appropriate. "
    "Explain which visible cues support your valence and arousal estimates and how they evolve over time. "
    "Avoid assumptions beyond what is visually present. "
    "Write a single coherent paragraph with no line breaks, no bullets, no special formatting, no longer than 100 tokens, "
    "and always end with a period."
)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def normalize_output(text: str) -> str:
    t = " ".join(text.replace("\n", " ").split()).strip()
    if t and not t.endswith("."):
        t += "."
    return t


def print_device_info(model) -> None:
    device_map = getattr(model, "hf_device_map", None)
    if isinstance(device_map, dict) and len(device_map) > 0:
        devices = sorted(set(device_map.values()))
        if any("cuda" in str(d) for d in devices):
            print(f"✅ Model device_map detected. Running on GPU(s): {devices}")
        else:
            print(f"✅ Model device_map detected. Running on: {devices}")
        return
    try:
        dev = next(model.parameters()).device
        if dev.type == "cuda":
            print(
                f"✅ Running on GPU: {torch.cuda.get_device_name(dev.index)} (cuda:{dev.index})"
            )
        else:
            print(f"✅ Running on CPU: {dev}")
    except StopIteration:
        print("⚠️ Could not determine device (model has no parameters?).")


def load_processed_video_names(output_csv: Path) -> Set[str]:
    if not output_csv.exists():
        return set()
    try:
        out_df = pl.read_csv(output_csv)
        if "video_name" in out_df.columns:
            return set(out_df["video_name"].to_list())
    except Exception:
        pass
    return set()


def vec_sanity_stats(x: np.ndarray) -> Tuple[float, float]:
    """Scalar mean/std across vector dimensions (NOT token-wise). Useful as sanity check."""
    return float(x.mean()), float(x.std())


def to_numpy(t: torch.Tensor, dtype: str) -> np.ndarray:
    t = t.detach().to("cpu")
    if dtype == "float16":
        t = t.to(torch.float16)
    else:
        t = t.to(torch.float32)
    return t.numpy()


def safe_write_pickle(path: Path, obj: Any) -> None:
    """Atomic-ish write: write temp then replace."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "wb") as f:
        pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL)
    tmp.replace(path)


def read_pickle_if_exists(path: Path, default: Any) -> Any:
    if not path.exists():
        return default
    with open(path, "rb") as f:
        return pickle.load(f)


# -------------------------
# Token-wise pooling
# -------------------------
def pooled_token_stats(
    hidden: torch.Tensor,  # [B,S,H]
    token_mask: torch.Tensor,  # [B,S] bool or 0/1
    eps: float = 1e-6,
) -> Dict[str, torch.Tensor]:
    """
    Returns dict:
      mean: [B,H] token-wise mean
      std:  [B,H] token-wise std (sqrt(E[x^2]-E[x]^2))
      last: [B,H] last token among token_mask==1 (fallback to last nonpad if needed)
    """
    if token_mask.dtype != torch.bool:
        token_mask_bool = token_mask > 0
    else:
        token_mask_bool = token_mask

    mask = token_mask_bool.unsqueeze(-1).to(hidden.dtype)  # [B,S,1]
    denom = mask.sum(dim=1).clamp_min(1.0)  # [B,1]

    mean = (hidden * mask).sum(dim=1) / denom  # [B,H]
    second = ((hidden**2) * mask).sum(dim=1) / denom
    var = (second - mean**2).clamp_min(0.0)
    std = torch.sqrt(var + eps)  # [B,H]

    # last: last index where token_mask_bool==True
    # if none True -> fallback to last token in sequence
    last_vecs = []
    B, S, H = hidden.shape
    for b in range(B):
        idx = torch.where(token_mask_bool[b])[0]
        if idx.numel() == 0:
            last_vecs.append(hidden[b, -1, :])
        else:
            last_vecs.append(hidden[b, idx[-1], :])
    last = torch.stack(last_vecs, dim=0)  # [B,H]

    return {"mean": mean, "std": std, "last": last}


@dataclass
class InferenceEngine:
    config: Config
    model: Optional[Qwen3VLForConditionalGeneration] = None
    processor: Optional[AutoProcessor] = None

    def __post_init__(self):
        self.model = Qwen3VLForConditionalGeneration.from_pretrained(
            self.config.model_name,
            dtype=self.config.torch_dtype,
            device_map="auto",
        )
        self.processor = AutoProcessor.from_pretrained(self.config.model_name)
        self.model.eval()

        self.config.embeddings_dir.mkdir(parents=True, exist_ok=True)
        if self.config.pickle_mode == "single":
            self.config.master_pickle_path.parent.mkdir(parents=True, exist_ok=True)

        print_device_info(self.model)

    # -------- message builders --------
    def build_messages_video_text(
        self, video_path: str, text_prompt: str
    ) -> List[Dict]:
        return [
            {
                "role": "user",
                "content": [
                    {
                        "type": "video",
                        "video": video_path,
                        "fps": self.config.video_fps,
                        "max_frames": self.config.max_frames,
                    },
                    {"type": "text", "text": text_prompt},
                ],
            }
        ]

    def build_messages_text_only(self, text_prompt: str) -> List[Dict]:
        return [{"role": "user", "content": [{"type": "text", "text": text_prompt}]}]

    # -------- inputs prep --------
    def _prepare_inputs_from_messages(
        self, messages: List[Dict]
    ) -> Dict[str, torch.Tensor]:
        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

        # Detect if vision present
        has_vision = any(
            c.get("type") in ("video", "image")
            for m in messages
            for c in m.get("content", [])
        )

        image_inputs = None
        video_inputs = None
        video_kwargs = {}
        video_metadatas = None

        if has_vision:
            try:
                image_inputs, video_inputs, video_kwargs = process_vision_info(
                    messages, return_video_kwargs=True, return_video_metadata=True
                )
                if (
                    video_inputs is not None
                    and len(video_inputs) > 0
                    and isinstance(video_inputs[0], tuple)
                ):
                    video_inputs, video_metadatas = zip(*video_inputs)
                    video_inputs, video_metadatas = list(video_inputs), list(
                        video_metadatas
                    )
            except TypeError:
                image_inputs, video_inputs, video_kwargs = process_vision_info(
                    messages, return_video_kwargs=True
                )
                video_metadatas = None

        proc_kwargs = dict(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
            **video_kwargs,
        )
        if video_metadatas is not None:
            proc_kwargs["video_metadata"] = video_metadatas

        inputs = self.processor(**proc_kwargs)
        inputs = {
            k: (v.to(self.model.device) if torch.is_tensor(v) else v)
            for k, v in inputs.items()
        }
        return inputs

    # -------- forward hidden --------
    def _forward_last_hidden(
        self, inputs: Dict[str, torch.Tensor]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        with torch.inference_mode():
            out = self.model(
                **inputs,
                output_hidden_states=True,
                use_cache=False,
                return_dict=True,
            )
        last_hidden = out.hidden_states[-1]  # [B,S,H]
        attn = inputs.get("attention_mask", None)
        if attn is None:
            attn = torch.ones(
                last_hidden.shape[:2], device=last_hidden.device, dtype=torch.long
            )
        return last_hidden, attn

    # -------- best-effort: visual token ids --------
    def _try_get_visual_placeholder_ids(self) -> List[int]:
        tok = getattr(self.processor, "tokenizer", None)
        if tok is None:
            return []

        candidates = [
            "<video>",
            "<image>",
            "<|video|>",
            "<|image|>",
            "<|vision_start|>",
            "<|vision_end|>",
            "<|video_start|>",
            "<|video_end|>",
            "<|image_start|>",
            "<|image_end|>",
        ]

        ids: List[int] = []
        for t in candidates:
            try:
                tid = tok.convert_tokens_to_ids(t)
                if isinstance(tid, int) and tid >= 0 and tid != tok.unk_token_id:
                    ids.append(tid)
            except Exception:
                pass

        for attr in ["image_token_id", "video_token_id", "vision_token_id"]:
            if hasattr(tok, attr):
                try:
                    tid = int(getattr(tok, attr))
                    if tid >= 0:
                        ids.append(tid)
                except Exception:
                    pass

        return sorted(set(ids))

    def _build_visual_mask_from_input_ids(
        self,
        input_ids: torch.Tensor,  # [B,S]
        attn: torch.Tensor,  # [B,S]
    ) -> Optional[torch.Tensor]:
        ids = self._try_get_visual_placeholder_ids()
        if not ids:
            return None
        vis_mask = torch.zeros_like(input_ids, dtype=torch.bool)
        for tid in ids:
            vis_mask |= input_ids == tid
        vis_mask &= attn > 0
        if vis_mask.sum().item() == 0:
            return None
        return vis_mask

    # -------- embeddings extraction (mean/std/last for each type) --------
    def extract_embeddings_pack(
        self, video_path: str
    ) -> Dict[str, Dict[str, np.ndarray]]:
        """
        Returns:
          {
            "multimodal": {"mean": vec, "std": vec, "last": vec},
            "text":       {"mean": vec, "std": vec, "last": vec},
            "visual":     {"mean": vec, "std": vec, "last": vec},
          }
        Where mean/std/last are token-wise pooled vectors.
        """

        # 1) multimodal: video + PROMPT_TEMPLATE
        mm_messages = self.build_messages_video_text(video_path, PROMPT_TEMPLATE)
        mm_inputs = self._prepare_inputs_from_messages(mm_messages)
        mm_hidden, mm_attn = self._forward_last_hidden(mm_inputs)

        mm_stats_t = pooled_token_stats(
            mm_hidden, mm_attn
        )  # token-wise over full sequence

        # 2) text-only: PROMPT_TEMPLATE only
        tx_messages = self.build_messages_text_only(PROMPT_TEMPLATE)
        tx_inputs = self._prepare_inputs_from_messages(tx_messages)
        tx_hidden, tx_attn = self._forward_last_hidden(tx_inputs)

        tx_stats_t = pooled_token_stats(tx_hidden, tx_attn)

        # 3) visual: try isolate visual placeholder tokens inside multimodal input_ids
        visual_stats_t: Optional[Dict[str, torch.Tensor]] = None
        try:
            if "input_ids" in mm_inputs:
                vis_mask = self._build_visual_mask_from_input_ids(
                    mm_inputs["input_ids"], mm_attn
                )
                if vis_mask is not None:
                    visual_stats_t = pooled_token_stats(mm_hidden, vis_mask)
        except Exception:
            visual_stats_t = None

        # fallback: video + anchor text (then token-wise over full sequence)
        if visual_stats_t is None:
            v_messages = self.build_messages_video_text(
                video_path, self.config.visual_anchor_text
            )
            v_inputs = self._prepare_inputs_from_messages(v_messages)
            v_hidden, v_attn = self._forward_last_hidden(v_inputs)
            visual_stats_t = pooled_token_stats(v_hidden, v_attn)

        out = {
            "multimodal": {
                "mean": to_numpy(mm_stats_t["mean"][0], self.config.emb_dtype),
                "std": to_numpy(mm_stats_t["std"][0], self.config.emb_dtype),
                "last": to_numpy(mm_stats_t["last"][0], self.config.emb_dtype),
            },
            "text": {
                "mean": to_numpy(tx_stats_t["mean"][0], self.config.emb_dtype),
                "std": to_numpy(tx_stats_t["std"][0], self.config.emb_dtype),
                "last": to_numpy(tx_stats_t["last"][0], self.config.emb_dtype),
            },
            "visual": {
                "mean": to_numpy(visual_stats_t["mean"][0], self.config.emb_dtype),
                "std": to_numpy(visual_stats_t["std"][0], self.config.emb_dtype),
                "last": to_numpy(visual_stats_t["last"][0], self.config.emb_dtype),
            },
        }
        return out

    # -------- text generation --------
    def generate_text(self, video_path: str) -> str:
        messages = self.build_messages_video_text(video_path, PROMPT_TEMPLATE)
        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

        try:
            image_inputs, video_inputs, video_kwargs = process_vision_info(
                messages, return_video_kwargs=True, return_video_metadata=True
            )
            video_metadatas = None
            if (
                video_inputs is not None
                and len(video_inputs) > 0
                and isinstance(video_inputs[0], tuple)
            ):
                video_inputs, video_metadatas = zip(*video_inputs)
                video_inputs, video_metadatas = list(video_inputs), list(
                    video_metadatas
                )
        except TypeError:
            image_inputs, video_inputs, video_kwargs = process_vision_info(
                messages, return_video_kwargs=True
            )
            video_metadatas = None

        proc_kwargs = dict(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
            **video_kwargs,
        )
        if video_metadatas is not None:
            proc_kwargs["video_metadata"] = video_metadatas

        inputs = self.processor(**proc_kwargs)
        inputs = {
            k: (v.to(self.model.device) if torch.is_tensor(v) else v)
            for k, v in inputs.items()
        }

        with torch.inference_mode():
            generated_ids = self.model.generate(
                **inputs,
                max_new_tokens=self.config.max_new_tokens,
                do_sample=False,
            )

        generated_ids_trimmed = [
            out_ids[len(in_ids) :]
            for in_ids, out_ids in zip(inputs["input_ids"], generated_ids)
        ]
        output_text = self.processor.batch_decode(
            generated_ids_trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )[0]
        return normalize_output(output_text)

    # -------- pickle saving --------
    def save_payload_pickle(self, video_name: str, payload: Dict[str, Any]) -> str:
        """
        payload will include embeddings + generated text.
        Returns a reference string (path).
        """
        if self.config.pickle_mode == "per_video":
            p = self.config.embeddings_dir / f"{video_name}.pkl"
            safe_write_pickle(p, payload)
            return str(p)

        # single master
        master = read_pickle_if_exists(self.config.master_pickle_path, default={})
        master[video_name] = payload
        safe_write_pickle(self.config.master_pickle_path, master)
        return str(self.config.master_pickle_path)


def main() -> None:
    cfg = Config()
    set_seed(cfg.seed)

    engine = InferenceEngine(cfg)
    df = pl.read_csv(cfg.input_csv)

    valence_col = "valence"
    arousal_col = "arousal"

    # skip invalid labels
    df = df.filter((pl.col(valence_col) != -5) & (pl.col(arousal_col) != -5))

    # Ensure output columns
    if "text" not in df.columns:
        df = df.with_columns(pl.lit("").alias("text"))
    if "pickle_ref" not in df.columns:
        df = df.with_columns(pl.lit("").alias("pickle_ref"))

    # Optional sanity-stat columns (scalar mean/std of each pooled vector)
    sanity_cols = [
        "mm_mean_vec_mean",
        "mm_mean_vec_std",
        "mm_std_vec_mean",
        "mm_std_vec_std",
        "mm_last_vec_mean",
        "mm_last_vec_std",
        "tx_mean_vec_mean",
        "tx_mean_vec_std",
        "tx_std_vec_mean",
        "tx_std_vec_std",
        "tx_last_vec_mean",
        "tx_last_vec_std",
        "vs_mean_vec_mean",
        "vs_mean_vec_std",
        "vs_std_vec_mean",
        "vs_std_vec_std",
        "vs_last_vec_mean",
        "vs_last_vec_std",
    ]
    for c in sanity_cols:
        if c not in df.columns:
            df = df.with_columns(pl.lit(None).alias(c))

    if df.height == 0:
        print("✅ valence/arousal == -5.")
        return

    processed = load_processed_video_names(cfg.output_csv)

    new_rows: List[Dict] = []
    saved_count = 0

    pbar = tqdm(df.iter_rows(named=True), total=df.height)
    for row in pbar:
        video_name = row.get("video_name")
        if not video_name or video_name in processed:
            continue

        # safety
        if row.get(valence_col, -5) == -5 or row.get(arousal_col, -5) == -5:
            continue

        parent_dir = video_name.split("___", 1)[0]
        video_path = cfg.video_dir / parent_dir / video_name
        if not video_path.exists():
            pbar.set_postfix(missing=str(video_path))
            continue

        try:
            # 1) embeddings (mean/std/last) for multimodal/text/visual
            emb_pack = engine.extract_embeddings_pack(str(video_path))

            # 2) generated text
            gen_text = engine.generate_text(str(video_path))

            payload = {
                "video_name": video_name,
                "video_path": str(video_path),
                "prompt": PROMPT_TEMPLATE,
                "video_fps": cfg.video_fps,
                "max_frames": cfg.max_frames,
                "emb_dtype": cfg.emb_dtype,
                "embeddings": emb_pack,  # dict[type][stat] = np.ndarray
                "text": gen_text,
            }

            pickle_ref = engine.save_payload_pickle(video_name, payload)

            # write into row
            row["text"] = gen_text
            row["pickle_ref"] = pickle_ref

            # sanity scalar stats in CSV (optional but handy)
            mm_mean_m, mm_mean_s = vec_sanity_stats(emb_pack["multimodal"]["mean"])
            mm_std_m, mm_std_s = vec_sanity_stats(emb_pack["multimodal"]["std"])
            mm_last_m, mm_last_s = vec_sanity_stats(emb_pack["multimodal"]["last"])

            tx_mean_m, tx_mean_s = vec_sanity_stats(emb_pack["text"]["mean"])
            tx_std_m, tx_std_s = vec_sanity_stats(emb_pack["text"]["std"])
            tx_last_m, tx_last_s = vec_sanity_stats(emb_pack["text"]["last"])

            vs_mean_m, vs_mean_s = vec_sanity_stats(emb_pack["visual"]["mean"])
            vs_std_m, vs_std_s = vec_sanity_stats(emb_pack["visual"]["std"])
            vs_last_m, vs_last_s = vec_sanity_stats(emb_pack["visual"]["last"])

            row["mm_mean_vec_mean"], row["mm_mean_vec_std"] = mm_mean_m, mm_mean_s
            row["mm_std_vec_mean"], row["mm_std_vec_std"] = mm_std_m, mm_std_s
            row["mm_last_vec_mean"], row["mm_last_vec_std"] = mm_last_m, mm_last_s

            row["tx_mean_vec_mean"], row["tx_mean_vec_std"] = tx_mean_m, tx_mean_s
            row["tx_std_vec_mean"], row["tx_std_vec_std"] = tx_std_m, tx_std_s
            row["tx_last_vec_mean"], row["tx_last_vec_std"] = tx_last_m, tx_last_s

            row["vs_mean_vec_mean"], row["vs_mean_vec_std"] = vs_mean_m, vs_mean_s
            row["vs_std_vec_mean"], row["vs_std_vec_std"] = vs_std_m, vs_std_s
            row["vs_last_vec_mean"], row["vs_last_vec_std"] = vs_last_m, vs_last_s

        except Exception as e:
            pbar.set_postfix(err=str(e), video=video_name)
            continue

        new_rows.append(row)
        processed.add(video_name)
        saved_count += 1

        # periodic save to CSV
        if saved_count % cfg.save_every == 0:
            if cfg.output_csv.exists():
                old = pl.read_csv(cfg.output_csv)
                merged = pl.concat(
                    [old, pl.DataFrame(new_rows)], how="diagonal_relaxed"
                )
            else:
                merged = pl.DataFrame(new_rows)
            merged = merged.unique(subset=["video_name"], keep="last")
            merged.write_csv(cfg.output_csv)
            new_rows.clear()

    # final flush
    if new_rows:
        if cfg.output_csv.exists():
            old = pl.read_csv(cfg.output_csv)
            merged = pl.concat([old, pl.DataFrame(new_rows)], how="diagonal_relaxed")
        else:
            merged = pl.DataFrame(new_rows)
        merged = merged.unique(subset=["video_name"], keep="last")
        merged.write_csv(cfg.output_csv)

    print(f"\n✅ CSV: {cfg.output_csv.absolute()}")
    if cfg.pickle_mode == "single":
        print(f"✅ Master pickle: {cfg.master_pickle_path.absolute()}")
        print("   dict[video_name] -> payload")
    else:
        print(f"✅ Per-video pickles: {cfg.embeddings_dir.absolute()}")
        print("   <video_name>.pkl -> payload")


if __name__ == "__main__":
    main()
