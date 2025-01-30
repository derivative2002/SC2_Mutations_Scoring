/**
 * API服务
 */

import axios, { AxiosError } from 'axios';
import type { ScoreRequest, ScoreResponse, RulesResponse } from '../types/api';

const api = axios.create({
  baseURL: 'http://localhost:8000/api',
  timeout: 5000,
  headers: {
    'Content-Type': 'application/json',
    'Accept': 'application/json',
  }
});

// 错误处理
const handleError = (error: unknown) => {
  console.log('API错误详情:', error);
  
  if (axios.isAxiosError(error)) {
    const axiosError = error as AxiosError<{ detail: string }>;
    console.log('请求配置:', axiosError.config);
    console.log('响应状态:', axiosError.response?.status);
    console.log('响应数据:', axiosError.response?.data);
    
    if (axiosError.response?.data?.detail) {
      throw new Error(axiosError.response.data.detail);
    }
    if (axiosError.code === 'ECONNABORTED') {
      throw new Error('请求超时，请重试');
    }
    if (!axiosError.response) {
      throw new Error('网络连接失败，请检查网络');
    }
    throw new Error(`请求失败: ${axiosError.message}`);
  }
  throw error;
};

export const getMaps = async (): Promise<string[]> => {
  try {
    console.log('获取地图列表...');
    const response = await api.get<string[]>('/mutations/maps');
    console.log('地图列表:', response.data);
    return response.data;
  } catch (error) {
    console.error('获取地图列表失败:', error);
    throw handleError(error);
  }
};

export const getCommanders = async (): Promise<string[]> => {
  try {
    console.log('获取指挥官列表...');
    const response = await api.get<string[]>('/mutations/commanders');
    console.log('指挥官列表:', response.data);
    return response.data;
  } catch (error) {
    console.error('获取指挥官列表失败:', error);
    throw handleError(error);
  }
};

export const getMutations = async (): Promise<string[]> => {
  try {
    console.log('获取突变因子列表...');
    const response = await api.get<string[]>('/mutations/mutations');
    console.log('突变因子列表:', response.data);
    return response.data;
  } catch (error) {
    console.error('获取突变因子列表失败:', error);
    throw handleError(error);
  }
};

export const getRules = async (): Promise<RulesResponse> => {
  try {
    console.log('获取规则列表...');
    const response = await api.get<RulesResponse>('/mutations/rules');
    console.log('规则列表:', response.data);
    return response.data;
  } catch (error) {
    console.error('获取规则列表失败:', error);
    throw handleError(error);
  }
};

export const scoreMutations = async (request: ScoreRequest): Promise<ScoreResponse> => {
  try {
    console.log('评分请求:', request);
    const response = await api.post<ScoreResponse>('/mutations/score', request);
    console.log('评分结果:', response.data);
    return response.data;
  } catch (error) {
    console.error('评分失败:', error);
    throw handleError(error);
  }
}; 