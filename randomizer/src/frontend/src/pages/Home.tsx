/**
 * 主页组件
 */

import React, { useState, useCallback } from 'react';
import { Layout, Typography, Space, Button, message } from 'antd';
import { ReloadOutlined } from '@ant-design/icons';
import styled from 'styled-components';
import { colors } from '../styles/theme';
import DifficultySelector from '../components/DifficultySelector';
import MutationResult from '../components/MutationResult';
import type { MutationCombination } from '../types/api';
import * as api from '../services/api';

const { Content } = Layout;
const { Title } = Typography;

const StyledLayout = styled(Layout)`
  min-height: 100vh;
  background: ${colors.background};
`;

const StyledContent = styled(Content)`
  max-width: 1200px;
  margin: 0 auto;
  padding: 40px 20px;
`;

const HeaderTitle = styled(Title)`
  text-align: center;
  color: ${colors.text.primary} !important;
  margin-bottom: 40px !important;
  text-shadow: 0 0 20px ${colors.primary}40;
`;

const GenerateButton = styled(Button)`
  min-width: 200px;
  height: 48px;
  font-size: 18px;
  margin-top: 40px;
`;

const MAX_RETRIES = 3;
const RETRY_DELAY = 500; // 毫秒

const Home: React.FC = () => {
  const [loading, setLoading] = useState(false);
  const [difficulty, setDifficulty] = useState(3);
  const [result, setResult] = useState<MutationCombination | null>(null);
  const [retryCount, setRetryCount] = useState(0);

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
        message.error('获取数据失败，请检查网络连接');
        throw error;
      });

      if (!maps.length || !commanders.length || !mutations.length) {
        message.error('获取到的数据为空，请检查后端服务');
        return;
      }

      // 随机选择地图和指挥官
      const map = maps[Math.floor(Math.random() * maps.length)];
      const selectedCommanders = [];
      const commanderCount = Math.random() < 0.7 ? 2 : 1; // 70%概率选择2个指挥官
      
      for (let i = 0; i < commanderCount; i++) {
        const availableCommanders = commanders.filter(c => !selectedCommanders.includes(c));
        if (availableCommanders.length === 0) break;
        const commander = availableCommanders[Math.floor(Math.random() * availableCommanders.length)];
        selectedCommanders.push(commander);
      }

      // 获取互斥规则
      const incompatiblePairs = rules.incompatible_pairs.map(
        rule => [rule.mutation1, rule.mutation2] as [string, string]
      );

      // 随机选择突变因子
      const selectedMutations = [];
      const mutationCount = Math.floor(Math.random() * 3) + 2; // 2-4个突变因子
      
      for (let i = 0; i < mutationCount; i++) {
        const availableMutations = mutations.filter(m => {
          // 检查是否与已选突变因子冲突
          for (const selected of selectedMutations) {
            if (incompatiblePairs.some(pair => 
              (pair[0] === m && pair[1] === selected) ||
              (pair[1] === m && pair[0] === selected)
            )) {
              return false;
            }
          }
          return !selectedMutations.includes(m);
        });
        
        if (availableMutations.length === 0) break;
        const mutation = availableMutations[Math.floor(Math.random() * availableMutations.length)];
        selectedMutations.push(mutation);
      }

      if (selectedMutations.length < 2) {
        throw new Error('无法生成足够的突变因子组合');
      }

      // 评分
      const score = await api.scoreMutations({
        map_name: map,
        commanders: selectedCommanders,
        mutations: selectedMutations,
      });

      // 如果分数与目标难度相差太大，重试
      if (Math.abs(score.score - difficulty) > 0.8) {
        if (retryAttempt < MAX_RETRIES) {
          console.log(`分数 ${score.score} 与目标难度 ${difficulty} 相差太大，重试第 ${retryAttempt + 1} 次`);
          await new Promise(resolve => setTimeout(resolve, RETRY_DELAY));
          setRetryCount(retryAttempt + 1);
          await generateMutation(retryAttempt + 1);
          return;
        } else {
          message.warning('已达到最大重试次数，显示最后一次生成结果');
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
        map,
        commanders: selectedCommanders,
        mutations: selectedMutations,
        difficulty: score.score,
        rules: ruleDescriptions,
      });
      
      setRetryCount(0);
      
    } catch (error) {
      console.error('生成失败:', error);
      if (retryAttempt < MAX_RETRIES) {
        console.log(`生成失败，重试第 ${retryAttempt + 1} 次`);
        await new Promise(resolve => setTimeout(resolve, RETRY_DELAY));
        setRetryCount(retryAttempt + 1);
        await generateMutation(retryAttempt + 1);
      } else {
        message.error('生成失败，请检查网络连接或刷新页面重试');
      }
    } finally {
      setLoading(false);
    }
  }, [difficulty]);

  return (
    <StyledLayout>
      <StyledContent>
        <HeaderTitle level={2}>
          星际争霸II 合作任务突变组合生成器
        </HeaderTitle>

        <Space direction="vertical" size="large" style={{ width: '100%' }}>
          <DifficultySelector
            value={difficulty}
            onChange={setDifficulty}
          />

          <GenerateButton
            type="primary"
            icon={<ReloadOutlined />}
            loading={loading}
            onClick={() => generateMutation(0)}
          >
            {loading ? `生成中${retryCount ? `(${retryCount}/${MAX_RETRIES})` : ''}` : '生成突变组合'}
          </GenerateButton>

          {result && <MutationResult combination={result} />}
        </Space>
      </StyledContent>
    </StyledLayout>
  );
};

export default Home; 