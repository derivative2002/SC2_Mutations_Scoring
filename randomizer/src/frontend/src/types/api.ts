/**
 * API类型定义
 */

export interface ScoreRequest {
  map_name: string;
  commanders: string[];
  mutations: string[];
  ai_type?: string;
}

export interface ScoreResponse {
  score: number;
  details: {
    rules: string[];
    num_mutations: number;
    commander_count: number;
  };
}

export interface MutationRule {
  mutation1: string;
  mutation2: string;
  description: string;
}

export interface RequiredPairRule {
  prerequisite: string;
  dependent: string;
  description: string;
}

export interface RulesResponse {
  incompatible_pairs: MutationRule[];
  required_pairs: RequiredPairRule[];
}

export interface MutationCombination {
  map: string;
  commanders: string[];
  mutations: string[];
  difficulty: number;
  rules: string[];
} 