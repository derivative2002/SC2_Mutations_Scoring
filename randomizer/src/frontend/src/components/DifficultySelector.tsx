/**
 * 难度选择器组件
 */

import React from 'react';
import { Card, Slider, Typography, Space } from 'antd';
import { RocketOutlined } from '@ant-design/icons';
import styled from 'styled-components';
import { colors } from '../styles/theme';

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

const DifficultyIcon = styled.div<{ difficulty: number }>`
  font-size: 24px;
  color: ${props => {
    if (props.difficulty <= 2) return colors.status.success;
    if (props.difficulty <= 3.5) return colors.status.warning;
    return colors.status.error;
  }};
`;

const DifficultyText = styled(Text)<{ difficulty: number }>`
  color: ${props => {
    if (props.difficulty <= 2) return colors.status.success;
    if (props.difficulty <= 3.5) return colors.status.warning;
    return colors.status.error;
  }};
  font-size: 16px;
  font-weight: bold;
`;

interface DifficultySelectorProps {
  value: number;
  onChange: (value: number) => void;
}

const getDifficultyText = (difficulty: number): string => {
  if (difficulty <= 1.5) return '轻松';
  if (difficulty <= 2.5) return '普通';
  if (difficulty <= 3.5) return '困难';
  if (difficulty <= 4.5) return '残酷';
  return '地狱';
};

const DifficultySelector: React.FC<DifficultySelectorProps> = ({ value, onChange }) => {
  return (
    <StyledCard
      title={
        <Space>
          <RocketOutlined />
          <Title level={4} style={{ margin: 0 }}>选择目标难度</Title>
        </Space>
      }
    >
      <Space direction="vertical" style={{ width: '100%' }} size="large">
        <div style={{ padding: '0 20px' }}>
          <Slider
            min={1}
            max={5}
            step={0.5}
            value={value}
            onChange={onChange}
            tooltip={{
              formatter: (value) => value ? `${value}分` : '',
            }}
          />
        </div>
        
        <Space style={{ justifyContent: 'center', width: '100%' }}>
          <DifficultyIcon difficulty={value}>
            <RocketOutlined />
          </DifficultyIcon>
          <DifficultyText difficulty={value}>
            {getDifficultyText(value)} ({value}分)
          </DifficultyText>
        </Space>
      </Space>
    </StyledCard>
  );
};

export default DifficultySelector; 