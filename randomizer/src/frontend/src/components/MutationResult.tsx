/**
 * 突变结果展示组件
 */

import React from 'react';
import { Card, Typography, Space, Tag, Divider } from 'antd';
import {
  RocketOutlined,
  TeamOutlined,
  EnvironmentOutlined,
  ExperimentOutlined,
  InfoCircleOutlined,
} from '@ant-design/icons';
import styled from 'styled-components';
import { colors } from '../styles/theme';
import type { MutationCombination } from '../types/api';

const { Title, Text } = Typography;

const StyledCard = styled(Card)`
  background: ${colors.surface};
  border: 1px solid ${colors.border};
  border-radius: 8px;
  box-shadow: 0 0 20px ${colors.primary}20;
  
  &:hover {
    box-shadow: 0 0 30px ${colors.primary}40;
  }

  .ant-card-head {
    border-bottom: 1px solid ${colors.border};
  }
`;

const StyledTag = styled(Tag)`
  font-size: 14px;
  padding: 4px 8px;
  border-radius: 4px;
  margin: 4px;
`;

const MapTag = styled(StyledTag)`
  background: ${colors.secondary}20;
  border-color: ${colors.secondary};
  color: ${colors.secondary};
`;

const CommanderTag = styled(StyledTag)`
  background: ${colors.primary}20;
  border-color: ${colors.primary};
  color: ${colors.primary};
`;

const MutationTag = styled(StyledTag)`
  background: ${colors.accent}20;
  border-color: ${colors.accent};
  color: ${colors.accent};
`;

const RuleText = styled(Text)`
  color: ${colors.text.secondary};
  font-size: 14px;
  line-height: 1.6;
  
  .anticon {
    color: ${colors.status.info};
    margin-right: 8px;
  }
`;

interface MutationResultProps {
  combination: MutationCombination;
}

const getDifficultyColor = (difficulty: number): string => {
  if (difficulty <= 2) return colors.status.success;
  if (difficulty <= 3.5) return colors.status.warning;
  return colors.status.error;
};

const MutationResult: React.FC<MutationResultProps> = ({ combination }) => {
  const { map, commanders, mutations, difficulty, rules } = combination;

  return (
    <StyledCard
      title={
        <Space>
          <RocketOutlined style={{ color: getDifficultyColor(difficulty) }} />
          <Title level={4} style={{ margin: 0 }}>
            难度 {difficulty.toFixed(1)} 分
          </Title>
        </Space>
      }
    >
      <Space direction="vertical" style={{ width: '100%' }} size="large">
        {/* 地图 */}
        <div>
          <Space align="center">
            <EnvironmentOutlined />
            <Text strong>地图</Text>
          </Space>
          <div style={{ marginTop: 8 }}>
            <MapTag>{map}</MapTag>
          </div>
        </div>

        <Divider style={{ margin: '12px 0' }} />

        {/* 指挥官 */}
        <div>
          <Space align="center">
            <TeamOutlined />
            <Text strong>指挥官</Text>
          </Space>
          <div style={{ marginTop: 8 }}>
            {commanders.map(commander => (
              <CommanderTag key={commander}>{commander}</CommanderTag>
            ))}
          </div>
        </div>

        <Divider style={{ margin: '12px 0' }} />

        {/* 突变因子 */}
        <div>
          <Space align="center">
            <ExperimentOutlined />
            <Text strong>突变因子</Text>
          </Space>
          <div style={{ marginTop: 8 }}>
            {mutations.map(mutation => (
              <MutationTag key={mutation}>{mutation}</MutationTag>
            ))}
          </div>
        </div>

        {/* 规则说明 */}
        {rules.length > 0 && (
          <>
            <Divider style={{ margin: '12px 0' }} />
            <div>
              <Space align="center" style={{ marginBottom: 8 }}>
                <InfoCircleOutlined />
                <Text strong>规则说明</Text>
              </Space>
              <Space direction="vertical" style={{ width: '100%' }}>
                {rules.map((rule, index) => (
                  <RuleText key={index}>
                    <InfoCircleOutlined />
                    {rule}
                  </RuleText>
                ))}
              </Space>
            </div>
          </>
        )}
      </Space>
    </StyledCard>
  );
};

export default MutationResult; 