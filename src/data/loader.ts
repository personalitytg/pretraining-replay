import type { AttentionAnnotation, AttentionData, CheckpointData, Discovery, EmbeddingData, Manifest, TokensOfInterest } from '../types';

const DATA_BASE = 'data';

function stepToFilename(step: number): string {
  return `step_${String(step).padStart(6, '0')}.json`;
}

async function getJson<T>(path: string): Promise<T> {
  const res = await fetch(path);
  if (!res.ok) {
    throw new Error(`fetch ${path} failed: ${res.status} ${res.statusText}`);
  }
  return (await res.json()) as T;
}

export function fetchManifest(): Promise<Manifest> {
  return getJson<Manifest>(`${DATA_BASE}/manifest.json`);
}

export function fetchCheckpoint(step: number): Promise<CheckpointData> {
  return getJson<CheckpointData>(`${DATA_BASE}/checkpoints/${stepToFilename(step)}`);
}

export function fetchDiscoveries(): Promise<Discovery[]> {
  return getJson<Discovery[]>(`${DATA_BASE}/discoveries.json`);
}

export async function fetchAttention(step: number): Promise<AttentionData> {
  const padded = step.toString().padStart(6, '0');
  const url = `${DATA_BASE}/attention/step_${padded}.bin`;
  const res = await fetch(url);
  if (!res.ok) throw new Error(`attention ${step}: ${res.status}`);
  const buf = await res.arrayBuffer();
  if (buf.byteLength < 8) throw new Error(`attention ${step}: file too short`);
  const view = new DataView(buf);
  const magic = String.fromCharCode(view.getUint8(0), view.getUint8(1), view.getUint8(2), view.getUint8(3));
  if (magic !== 'ATTN') throw new Error(`attention ${step}: bad magic ${magic}`);
  const n_layer = view.getUint8(4);
  const n_head = view.getUint8(5);
  const seq_len = view.getUint8(6);
  const expected = n_layer * n_head * seq_len * seq_len;
  if (buf.byteLength - 8 !== expected) {
    throw new Error(`attention ${step}: body size ${buf.byteLength - 8} != ${expected}`);
  }
  const int8 = new Int8Array(buf, 8, expected);
  const values = new Float32Array(expected);
  for (let i = 0; i < expected; i++) values[i] = int8[i] / 127;
  return { step, n_layer, n_head, seq_len, values };
}

export async function fetchEmbedding(step: number): Promise<EmbeddingData> {
  const padded = step.toString().padStart(6, '0');
  const url = `${DATA_BASE}/embeddings/step_${padded}.bin`;
  const res = await fetch(url);
  if (!res.ok) throw new Error(`embedding ${step}: ${res.status}`);
  const buf = await res.arrayBuffer();
  if (buf.byteLength < 16) throw new Error(`embedding ${step}: file too short`);
  const view = new DataView(buf);
  const magic = String.fromCharCode(view.getUint8(0), view.getUint8(1), view.getUint8(2), view.getUint8(3));
  if (magic !== 'EMB1') throw new Error(`embedding ${step}: bad magic ${magic}`);
  const n_tokens = view.getInt32(4, true);
  const n_dims = view.getInt32(8, true);
  const expected = n_tokens * n_dims;
  if (buf.byteLength - 16 !== expected * 4) {
    throw new Error(`embedding ${step}: body ${buf.byteLength - 16} != ${expected * 4}`);
  }
  const coords = new Float32Array(buf.slice(16));
  return { step, n_tokens, n_dims, coords };
}

export async function fetchAttentionAnnotations(): Promise<AttentionAnnotation[]> {
  const res = await fetch(`${DATA_BASE}/attention_annotations.json`);
  if (!res.ok) throw new Error(`attention_annotations: ${res.status}`);
  return res.json();
}

export async function fetchTokensOfInterest(): Promise<TokensOfInterest> {
  const res = await fetch(`${DATA_BASE}/tokens_of_interest.json`);
  if (!res.ok) throw new Error(`tokens_of_interest: ${res.status}`);
  return res.json();
}
