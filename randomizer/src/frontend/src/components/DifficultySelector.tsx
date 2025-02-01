/**
 * 难度选择器组件
 */

import React from 'react';
import { Card, Slider, Typography, Space } from '@arco-design/web-react';
import { IconFire } from '@arco-design/web-react/icon';
import styled from 'styled-components';
import { colors } from '../styles/theme';

const { Title, Text } = Typography;

const StyledCard = styled(Card)`
  background: ${colors.surface};
  border: 1px solid ${colors.border};
  border-radius: 16px;
  box-shadow: 0 4px 20px ${colors.primary}20;
  
  &:hover {
    box-shadow: 0 8px 30px ${colors.primary}40;
  }

  .arco-card-header {
    border-bottom: 1px solid ${colors.border};
    padding: 20px 24px;
  }

  .arco-card-body {
    padding: 24px;
  }

  transition: all 0.3s ease;
`;

const DifficultyIcon = styled.div<{ difficulty: number }>`
  font-size: 28px;
  color: ${props => {
    if (props.difficulty <= 2) return colors.status.success;
    if (props.difficulty <= 3.5) return colors.status.warning;
    return colors.status.error;
  }};
  display: flex;
  align-items: center;
  justify-content: center;
  width: 48px;
  height: 48px;
  border-radius: 24px;
  background: ${props => {
    if (props.difficulty <= 2) return `${colors.status.success}20`;
    if (props.difficulty <= 3.5) return `${colors.status.warning}20`;
    return `${colors.status.error}20`;
  }};
`;

const DifficultyText = styled(Text)<{ difficulty: number }>`
  color: ${props => {
    if (props.difficulty <= 2) return colors.status.success;
    if (props.difficulty <= 3.5) return colors.status.warning;
    return colors.status.error;
  }};
  font-size: 18px;
  font-weight: bold;
`;

interface DifficultySelectorProps {
  value: number;
  onChange: (value: number | number[]) => void;
}

const getDifficultyText = (difficulty: number): string => {
  if (difficulty <= 1.5) return '轻松';
  if (difficulty <= 2.5) return '普通';
  if (difficulty <= 3.5) return '困难';
  if (difficulty <= 4.5) return '残酷';
  return '地狱';
};

const DifficultySelector: React.FC<DifficultySelectorProps> = ({ value, onChange }) => {
  const handleChange = (val: number | number[]) => {
    onChange(typeof val === 'number' ? val : val[0]);
  };

  return (
    <StyledCard
      title={
        <Space size="large">
          <IconFire style={{ fontSize: '24px' }} />
          <Title heading={4} style={{ margin: 0, fontSize: '20px' }}>选择目标难度</Title>
        </Space>
      }
    >
      <Space direction="vertical" size="large" style={{ width: '100%' }}>
        <div style={{ padding: '0 20px' }}>
          <Slider
            min={1}
            max={5}
            step={1}
            value={value}
            onChange={handleChange}
            formatTooltip={(val) => `${val}分`}
            marks={{
              1: '1',
              2: '2',
              3: '3',
              4: '4',
              5: '5'
            }}
            style={{ margin: '20px 0' }}
            showTicks
          />
        </div>
        
        <Space style={{ justifyContent: 'center', width: '100%' }} size="large">
          <DifficultyIcon difficulty={value}>
            <IconFire />
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