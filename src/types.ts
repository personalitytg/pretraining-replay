export interface ModelConfig {
  n_layer: number;
  n_head: number;
  d_model: number;
  d_ff: number;
  ctx_len: number;
  vocab_size: number;
  param_count: number;
}

export interface TrainingConfig {
  dataset: string;
  tokenizer: string;
  total_steps: number;
  batch_size_effective: number;
  tokens_per_step: number;
  total_tokens_seen: number;
  peak_lr: number;
  seed: number;
  canonical_input_tokens?: string[];
}

export interface CheckpointMeta {
  step: number;
  tokens_seen: number;
  wallclock_seconds: number;
  loss_train: number;
  loss_val: number;
  lr: number;
  sha256?: string | null;
}

export type PromptRole = 'showcase' | 'probe';

export interface Prompt {
  id: string;
  text: string;
  role?: PromptRole;
}

export type ProbeKind = 'fraction' | 'number' | 'boolean';

export interface Probe {
  id: string;
  title: string;
  kind: ProbeKind;
  threshold: number | null;
}

export interface VizSettings {
  embedding_xrange: [number, number];
  embedding_yrange: [number, number];
}

export interface Manifest {
  version: string;
  run_id: string;
  model: ModelConfig;
  training: TrainingConfig;
  checkpoints: CheckpointMeta[];
  prompts: Prompt[];
  probes: Probe[];
  probe_sparklines: Record<string, Array<number | boolean>>;
  tokens_of_interest_count: number;
  viz_settings: VizSettings;
}

export interface GenerationSample {
  seed: number;
  tokens: number[];
  text: string;
}

export interface TopFiveCandidate {
  token: string;
  prob: number;
}

export interface TopFiveEntry {
  position: number;
  context_token: string;
  top_5: TopFiveCandidate[];
}

export interface LayerNorms {
  layer: number;
  qkv_norm: number;
  attn_out_norm: number;
  ffn_up_norm: number;
  ffn_down_norm: number;
}

export interface WeightStats {
  embedding_norm: number;
  output_embedding_norm: number;
  layer_norms: LayerNorms[];
  embedding_top5_singular_values: number[];
  attention_logit_mean_per_layer?: number[];
  attention_logit_std_per_layer?: number[];
}

export interface ProbeResults {
  uses_quotation_marks: number;
  past_tense_after_yesterday: number;
  plural_after_two: number;
  coherent_three_word_phrase: number;
  multi_sentence: number;
  consistent_named_entity: number;
  vocabulary_size: number;
  loss_below_threshold: boolean;
}

export interface DiscoveryExample {
  prompt_id: string;
  seed: number;
  generation: string;
  step?: number;
}

export interface AttentionAnnotation {
  layer: number;
  head: number;
  form_step: number;
  title: string;
  description: string;
}

export interface Discovery {
  id: string;
  title: string;
  step: number;
  probe_id: string;
  headline_example: DiscoveryExample;
  before_example: DiscoveryExample;
  explanation: string;
  tweet_text: string;
}

export type Theme = 'dark' | 'light';

export type ViewType = 'text' | 'probes' | 'attention' | 'embedding' | 'predict';

export type ViewMode = 'interactive' | 'showcase';

export interface CheckpointData {
  step: number;
  generations: Record<string, GenerationSample[]>;
  top_5_next: TopFiveEntry[];
  weight_stats?: WeightStats;
  probe_results?: ProbeResults;
}

export interface AttentionData {
  step: number;
  n_layer: number;
  n_head: number;
  seq_len: number;
  // flat float32 в [0,1], length = n_layer * n_head * seq_len * seq_len
  values: Float32Array;
}

export interface EmbeddingData {
  step: number;
  n_tokens: number;
  n_dims: number;
  coords: Float32Array;
}

export interface TokenOfInterest {
  id: number;
  token_text: string;
  category: string;
}

export interface TokensOfInterest {
  tokens: TokenOfInterest[];
}
