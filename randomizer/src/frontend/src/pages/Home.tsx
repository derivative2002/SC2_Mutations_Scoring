/**
 * 主页组件
 */

import React, { useState, useCallback } from 'react';
import { Layout, Typography, Space, Button, Message } from '@arco-design/web-react';
import { IconRefresh, IconLeft } from '@arco-design/web-react/icon';
import styled from 'styled-components';
import { colors } from '../styles/theme';
import DifficultySelector from '../components/DifficultySelector';
import MutationResult from '../components/MutationResult';
import type { MutationCombination, MapInfo, CommanderInfo, MutationInfo } from '../types/api';
import * as api from '../services/api';

const { Content } = Layout;
const { Title } = Typography;

const StyledLayout = styled(Layout)`
  min-height: 100vh;
  background: ${colors.background};
  display: flex;
  justify-content: center;
  align-items: center;
  padding: 20px;
`;

const StyledContent = styled(Content)`
  width: 100%;
  max-width: 1600px;
  margin: 0 auto;
  height: calc(100vh - 40px);
  display: flex;
  flex-direction: column;
  justify-content: center;
`;

const HeaderTitle = styled(Title)`
  text-align: center;
  color: ${colors.text.primary} !important;
  margin-bottom: 60px !important;
  text-shadow: 0 0 20px ${colors.primary}40;
  font-size: 32px !important;
  font-weight: bold !important;
`;

const BackButton = styled(Button)`
  position: fixed;
  top: 20px;
  left: 20px;
  width: 48px;
  height: 48px;
  border-radius: 24px;
  display: flex;
  align-items: center;
  justify-content: center;
  font-size: 24px;
  background: ${colors.surface};
  border-color: ${colors.border};
  color: ${colors.text.primary};
  
  &:hover {
    background: ${colors.surfaceLight};
    transform: translateX(-2px);
  }
  
  transition: all 0.3s ease;
`;

const GenerateButton = styled(Button)`
  min-width: 240px;
  height: 52px;
  font-size: 20px;
  margin-top: 60px;
  border-radius: 8px;
  
  &:hover {
    transform: translateY(-2px);
    box-shadow: 0 4px 12px ${colors.primary}40;
  }
  
  transition: all 0.3s ease;
`;

const MAX_RETRIES = 3;
const RETRY_DELAY = 500; // 毫秒

const Home: React.FC = () => {
  const [loading, setLoading] = useState(false);
  const [difficulty, setDifficulty] = useState(3);  // 默认中等难度
  const [result, setResult] = useState<MutationCombination | null>(null);
  const [retryCount, setRetryCount] = useState(0);
  const [showResult, setShowResult] = useState(false);

  const handleDifficultyChange = useCallback((value: number | number[]) => {
    setDifficulty(typeof value === 'number' ? value : value[0]);
  }, []);

  const handleBack = useCallback(() => {
    setShowResult(false);
  }, []);

  const generateMutation = useCallback(async (retryAttempt = 0) => {
    try {
      setLoading(true);
      
      // 获取可用选项
      const [maps, commanders, mutations, rules] = await Promise.all([
        api.getMaps(),
        api.getCommanders(),
        api.getMutations(),
        api.getRules(),
      ]).catch(error => {
        console.error('获取数据失败:', error);
        Message.error('获取数据失败，请检查网络连接');
        throw error;
      });

      if (!maps.length || !commanders.length || !mutations.length) {
        Message.error('获取到的数据为空，请检查后端服务');
        return;
      }

      // 随机选择地图和指挥官
      const selectedMap = maps[Math.floor(Math.random() * maps.length)];
      console.log('选择的地图:', selectedMap);  // 调试日志
      const selectedCommanders: string[] = [];
      
      // 选择两个指挥官
      for (let i = 0; i < 2; i++) {
        const availableCommanders = commanders.filter(c => !selectedCommanders.includes(c.name));
        if (availableCommanders.length === 0) break;
        const commander = availableCommanders[Math.floor(Math.random() * availableCommanders.length)];
        console.log('选择的指挥官:', commander);  // 调试日志
        selectedCommanders.push(commander.name);
      }

      // 如果没有选择到两个指挥官，抛出错误
      if (selectedCommanders.length !== 2) {
        throw new Error('无法选择足够的指挥官');
      }

      // 获取互斥规则
      const incompatiblePairs = rules.incompatible_pairs.map(
        rule => [rule.mutation1, rule.mutation2] as [string, string]
      );

      // 获取排除的突变因子
      const excludedMutations = new Set(
        (rules.generation_rules?.excluded_mutations || []).map(m => m.name)
      );
      
      // 随机选择突变因子
      const selectedMutations: string[] = [];
      
      // 首先添加固定的突变因子
      const fixedMutations = rules.generation_rules?.fixed_mutations || [];
      for (const mutation of fixedMutations) {
        selectedMutations.push(mutation.name);
      }
      
      // 然后随机选择剩余的突变因子
      const mutationCount = rules.generation_rules?.mutation_count || { min: 2, max: 4 };
      const targetCount = Math.floor(Math.random() * (mutationCount.max - mutationCount.min + 1)) + mutationCount.min;
      const remainingCount = Math.max(0, targetCount - selectedMutations.length);
      
      // 创建权重表
      const weightedMutations = new Map<string, number>();
      (rules.generation_rules?.weighted_mutations || []).forEach(mutation => {
        weightedMutations.set(mutation.name, mutation.weight);
      });
      
      for (let i = 0; i < remainingCount; i++) {
        const availableMutations = mutations.filter(m => {
          // 检查是否在排除列表中
          if (excludedMutations.has(m.name)) {
            return false;
          }
          
          // 检查是否与已选突变因子冲突
          for (const selected of selectedMutations) {
            if (incompatiblePairs.some(pair => 
              (pair[0] === m.name && pair[1] === selected) ||
              (pair[1] === m.name && pair[0] === selected)
            )) {
              return false;
            }
          }
          return !selectedMutations.includes(m.name);
        });
        
        if (availableMutations.length === 0) break;
        
        // 根据权重选择突变因子
        const totalWeight = availableMutations.reduce((sum, m) => 
          sum + (weightedMutations.get(m.name) || 1.0), 0);
        let randomWeight = Math.random() * totalWeight;
        
        let selectedMutation = availableMutations[0];
        for (const mutation of availableMutations) {
          const weight = weightedMutations.get(mutation.name) || 1.0;
          if (randomWeight <= weight) {
            selectedMutation = mutation;
            break;
          }
          randomWeight -= weight;
        }
        
        console.log('选择的突变因子:', selectedMutation);  // 调试日志
        selectedMutations.push(selectedMutation.name);
      }

      if (selectedMutations.length < mutationCount.min) {
        throw new Error('无法生成足够的突变因子组合');
      }

      // 随机选择 AI 类型
      const aiTypes = [
        "帝国战斗群", "旧世机械团", "爆炸威胁", "暗影袭扰", "战争机械团",
        "袭扰炮击", "旧世步兵团", "侵略虫群", "滋生腐化", "艾尔先锋",
        "遮天蔽日", "步战机甲", "族长之军", "大师机械", "突击团",
        "暗影科技团", "肆虐扩散", "风暴迫临", "卡莱的希望"
      ];
      const selectedAiType = aiTypes[Math.floor(Math.random() * aiTypes.length)];
      console.log('选择的AI类型:', selectedAiType);  // 调试日志

      // 评分
      const scoreRequest = {
        map_name: selectedMap.name,
        commanders: selectedCommanders,
        mutations: selectedMutations,
        ai_type: selectedAiType  // 使用随机选择的 AI 类型
      };
      console.log('评分请求:', JSON.stringify(scoreRequest));  // 调试日志
      const score = await api.scoreMutations(scoreRequest);

      // 如果分数与目标难度不同，重试
      if (score.score !== Math.round(difficulty)) {
        if (retryAttempt < MAX_RETRIES) {
          console.log(`分数 ${score.score} 与目标难度 ${difficulty} 不匹配，重试第 ${retryAttempt + 1} 次`);
          await new Promise(resolve => setTimeout(resolve, RETRY_DELAY));
          setRetryCount(retryAttempt + 1);
          await generateMutation(retryAttempt + 1);
          return;
        } else {
          Message.warning('已达到最大重试次数，显示最后一次生成结果');
        }
      }

      // 获取规则说明
      const ruleDescriptions = [];
      for (let i = 0; i < selectedMutations.length; i++) {
        for (let j = i + 1; j < selectedMutations.length; j++) {
          const m1 = selectedMutations[i];
          const m2 = selectedMutations[j];
          
          // 检查互斥规则
          const incompatibleRule = rules.incompatible_pairs.find(
            rule => (rule.mutation1 === m1 && rule.mutation2 === m2) ||
                   (rule.mutation1 === m2 && rule.mutation2 === m1)
          );
          if (incompatibleRule) {
            ruleDescriptions.push(incompatibleRule.description);
          }
          
          // 检查依赖规则
          const requiredRule = rules.required_pairs.find(
            rule => (rule.prerequisite === m1 && rule.dependent === m2) ||
                   (rule.prerequisite === m2 && rule.dependent === m1)
          );
          if (requiredRule) {
            ruleDescriptions.push(requiredRule.description);
          }
        }
      }

      setResult({
        map: selectedMap.name,
        commanders: selectedCommanders,
        mutations: selectedMutations,
        difficulty: score.score,
        rules: ruleDescriptions,
        ai_type: scoreRequest.ai_type,
      });
      
      setRetryCount(0);
      setShowResult(true);
      
    } catch (error) {
      console.error('生成失败:', error);
      if (retryAttempt < MAX_RETRIES) {
        console.log(`生成失败，重试第 ${retryAttempt + 1} 次`);
        await new Promise(resolve => setTimeout(resolve, RETRY_DELAY));
        setRetryCount(retryAttempt + 1);
        await generateMutation(retryAttempt + 1);
      } else {
        Message.error('生成失败，请检查网络连接或刷新页面重试');
      }
    } finally {
      setLoading(false);
    }
  }, [difficulty]);

  return (
    <StyledLayout>
      <StyledContent>
        {showResult ? (
          <>
            <BackButton icon={<IconLeft />} onClick={handleBack} />
            {result && <MutationResult combination={result} />}
          </>
        ) : (
          <>
            <HeaderTitle heading={1}>星际争霸2 合作任务自定义挑战AI随机器</HeaderTitle>
            <Space direction="vertical" size="large" style={{ width: '100%', maxWidth: '800px', margin: '0 auto' }}>
              <DifficultySelector value={difficulty} onChange={handleDifficultyChange} />
              <div style={{ textAlign: 'center' }}>
                <GenerateButton
                  type="primary"
                  loading={loading}
                  onClick={() => generateMutation(0)}
                  icon={<IconRefresh />}
                >
                  生成突变组合
                </GenerateButton>
              </div>
            </Space>
          </>
        )}
      </StyledContent>
    </StyledLayout>
  );
};

export default Home; 