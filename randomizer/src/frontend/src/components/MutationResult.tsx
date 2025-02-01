/**
 * 突变结果展示组件
 */

import React from 'react';
import { Card, Typography, Space, Tag, Divider, Image, Grid, Skeleton } from '@arco-design/web-react';
import {
  IconFire,
  IconUserGroup,
  IconLocation,
  IconApps,
  IconInfo,
  IconImage,
} from '@arco-design/web-react/icon';
import styled from 'styled-components';
import { colors } from '../styles/theme';
import type { MutationCombination } from '../types/api';

const { Title, Text } = Typography;
const { Row, Col } = Grid;

const StyledCard = styled(Card)`
  background: ${colors.surface};
  border: 1px solid ${colors.border};
  border-radius: 16px;
  box-shadow: 0 4px 20px ${colors.primary}20;
  
  .arco-card-header {
    border-bottom: 1px solid ${colors.border};
    padding: 16px;
  }

  .arco-card-body {
    padding: 16px;
  }
`;

const ImagesCard = styled(Card)`
  background: ${colors.surfaceLight};
  border: 1px solid ${colors.border};
  border-radius: 12px;
`;

const ResultLayout = styled.div`
  display: grid;
  grid-template-columns: 400px 1fr;
  gap: 24px;
  width: 100%;
  height: 100%;
  max-width: 1200px;
  margin: 0 auto;
  max-height: calc(100vh - 40px);
`;

const LeftColumn = styled.div`
  display: flex;
  flex-direction: column;
  gap: 16px;
`;

const RightColumn = styled.div`
  display: flex;
  flex-direction: column;
  gap: 16px;
`;

const InfoTag = styled.div<{ type?: 'map' | 'commander' | 'ai' | 'mutation' }>`
  padding: 8px 12px;
  border-radius: 8px;
  background: ${props => {
    switch (props.type) {
      case 'map':
        return `${colors.secondary}15`;
      case 'commander':
        return `${colors.primary}15`;
      case 'ai':
        return `${colors.accent}15`;
      case 'mutation':
        return `${colors.status.warning}15`;
      default:
        return colors.surface;
    }
  }};
  border: 1px solid ${props => {
    switch (props.type) {
      case 'map':
        return colors.secondary;
      case 'commander':
        return colors.primary;
      case 'ai':
        return colors.accent;
      case 'mutation':
        return colors.status.warning;
      default:
        return colors.border;
    }
  }};
  color: ${colors.text.primary};
  margin: 4px 0;
  font-size: 14px;
`;

const InfoTagsContainer = styled.div`
  display: flex;
  flex-direction: column;
  gap: 8px;
`;

const TopImagesGrid = styled.div`
  display: grid;
  grid-template-columns: 1fr 1fr;
  gap: 24px;
  padding: 16px;
  border-bottom: 1px solid ${colors.border};
`;

const CommandersGrid = styled.div`
  display: grid;
  grid-template-columns: repeat(2, 1fr);
  gap: 16px;
  padding: 16px;
  border-bottom: 1px solid ${colors.border};
`;

const MutationsGrid = styled.div`
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(120px, 1fr));
  gap: 16px;
  padding: 16px;
`;

const ImageContainer = styled.div`
  display: flex;
  flex-direction: column;
  align-items: center;
  gap: 8px;
  text-align: center;
  width: 100%;
  padding: 8px;
  background: ${colors.surface};
  border-radius: 8px;
`;

const StyledImage = styled(Image)`
  width: 100%;
  height: auto;
  border-radius: 4px;

  .arco-image {
    width: 100%;
    height: auto;
  }

  .arco-image-img {
    object-fit: contain;
  }
`;

const MapImageContainer = styled(ImageContainer)`
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: center;
  padding: 12px;
  
  .arco-image {
    display: flex;
    align-items: center;
    justify-content: center;
    width: 100%;
    height: 100%;
  }

  .arco-image-img {
    object-fit: contain;
    max-width: 100%;
    max-height: 100%;
  }
`;

const MapImage = styled(StyledImage)`
  min-height: 200px;
  max-height: 280px;
  width: 100%;
  
  .arco-image-img {
    object-fit: contain;
  }
`;

const AIImageContainer = styled(ImageContainer)`
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: center;
  padding: 12px;
  
  .arco-image {
    display: flex;
    align-items: center;
    justify-content: center;
    width: 100%;
    height: 100%;
  }

  .arco-image-img {
    object-fit: contain;
    max-width: 100%;
    max-height: 100%;
  }
`;

const AIImage = styled(StyledImage)`
  min-height: 200px;
  max-height: 280px;
  width: 100%;
  
  .arco-image-img {
    object-fit: contain;
  }
`;

const CommanderImage = styled(StyledImage)`
  min-height: 120px;
  max-height: 180px;
`;

const MutationImage = styled(StyledImage)`
  min-height: 80px;
  max-height: 120px;
`;

const ImageLabel = styled(Text)`
  font-size: 14px;
  color: ${colors.text.primary};
  margin-top: 8px;
  white-space: nowrap;
  overflow: hidden;
  text-overflow: ellipsis;
  max-width: 100%;
`;

const InfoSection = styled.div`
  display: flex;
  flex-direction: column;
  gap: 8px;
`;

const InfoTitle = styled(Text)`
  font-size: 16px;
  color: ${colors.text.primary};
  font-weight: bold;
`;

const InfoValue = styled(Text)`
  font-size: 14px;
  color: ${colors.text.secondary};
`;

const DifficultyInfo = styled.div`
  display: flex;
  align-items: center;
  gap: 8px;
  margin-bottom: 8px;
`;

const DifficultyScore = styled(Text)<{ difficulty: number }>`
  font-size: 24px;
  font-weight: bold;
  color: ${props => {
    if (props.difficulty <= 2) return colors.status.success;
    if (props.difficulty <= 3.5) return colors.status.warning;
    return colors.status.error;
  }};
`;

const ImageErrorState = styled.div`
  background: ${colors.surface};
  width: 100%;
  height: 100%;
  min-height: inherit;
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: center;
  gap: 8px;
  color: ${colors.text.secondary};
  border-radius: 4px;
  border: 1px dashed ${colors.border};
  padding: 16px;
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
  const { map, commanders, mutations, difficulty = 0, rules = [], ai_type } = combination;
  const [loading, setLoading] = React.useState(true);

  const getImagePath = (name: string, type: string) => {
    switch (type) {
      case 'maps':
        return `/resources/images/maps/${name}_display.png`;
      case 'commanders':
        return `/resources/images/commanders/${name}_display.png`;
      case 'mutations':
        return `/resources/images/mutations/${name}_display.png`;
      case 'ai_types':
        return `/resources/images/ai_types/${name}_display.png`;
      default:
        return '';
    }
  };

  const handleImageError = (e: React.SyntheticEvent<HTMLImageElement, Event>, type: string, name: string) => {
    const target = e.target as HTMLImageElement;
    if (target.src.includes('_display')) {
      target.src = target.src.replace('_display.png', '.png');
    }
  };

  const handleImageLoad = () => {
    setLoading(false);
  };

  const renderErrorState = (height: number) => (
    <ImageErrorState style={{ minHeight: `${height}px` }}>
      <IconImage style={{ fontSize: 24 }} />
      <span>图片加载失败</span>
    </ImageErrorState>
  );

  return (
    <ResultLayout>
      <LeftColumn>
        <StyledCard>
          <DifficultyInfo>
            <IconFire style={{ fontSize: '24px', color: getDifficultyColor(difficulty || 0) }} />
            <DifficultyScore difficulty={difficulty || 0}>
              难度 {(difficulty || 0).toFixed(1)} 分
            </DifficultyScore>
          </DifficultyInfo>

          <InfoSection>
            <InfoTitle>地图</InfoTitle>
            <InfoTagsContainer>
              <InfoTag type="map">{map}</InfoTag>
            </InfoTagsContainer>
          </InfoSection>

          <InfoSection>
            <InfoTitle>指挥官</InfoTitle>
            <InfoTagsContainer>
              {commanders.map(commander => (
                <InfoTag key={commander} type="commander">{commander}</InfoTag>
              ))}
            </InfoTagsContainer>
          </InfoSection>

          <InfoSection>
            <InfoTitle>AI类型</InfoTitle>
            <InfoTagsContainer>
              <InfoTag type="ai">{ai_type}</InfoTag>
            </InfoTagsContainer>
          </InfoSection>

          <InfoSection>
            <InfoTitle>突变因子</InfoTitle>
            <InfoTagsContainer>
              {mutations.map(mutation => (
                <InfoTag key={mutation} type="mutation">{mutation}</InfoTag>
              ))}
            </InfoTagsContainer>
          </InfoSection>

          {rules.length > 0 && (
            <InfoSection>
              <InfoTitle>规则说明</InfoTitle>
              <InfoTagsContainer>
                {rules.map((rule, index) => (
                  <InfoValue key={index}>{rule}</InfoValue>
                ))}
              </InfoTagsContainer>
            </InfoSection>
          )}
        </StyledCard>
      </LeftColumn>

      <RightColumn>
        <ImagesCard>
          <Row gutter={24} style={{ padding: '16px', borderBottom: `1px solid ${colors.border}` }}>
            <Col span={12}>
              <MapImageContainer>
                <MapImage 
                  src={getImagePath(map, 'maps')}
                  alt={`地图：${map}`}
                  onLoad={handleImageLoad}
                  onError={(e) => handleImageError(e, 'maps', map)}
                  error={renderErrorState(200)}
                  loader={<Skeleton animation image style={{ width: '100%', height: '200px' }} />}
                  actions={[
                    <div key="preview" style={{ color: colors.text.primary }}>点击预览</div>
                  ]}
                />
                <ImageLabel>地图：{map}</ImageLabel>
              </MapImageContainer>
            </Col>
            <Col span={12}>
              <AIImageContainer>
                <AIImage 
                  src={getImagePath(ai_type, 'ai_types')}
                  alt={`AI：${ai_type}`}
                  onLoad={handleImageLoad}
                  onError={(e) => handleImageError(e, 'ai_types', ai_type)}
                  error={renderErrorState(200)}
                  loader={<Skeleton animation image style={{ width: '100%', height: '200px' }} />}
                  actions={[
                    <div key="preview" style={{ color: colors.text.primary }}>点击预览</div>
                  ]}
                />
                <ImageLabel>AI：{ai_type}</ImageLabel>
              </AIImageContainer>
            </Col>
          </Row>

          <Row gutter={16} style={{ padding: '16px', borderBottom: `1px solid ${colors.border}` }}>
            {commanders.map(commander => (
              <Col span={12} key={commander}>
                <ImageContainer>
                  <CommanderImage 
                    src={getImagePath(commander, 'commanders')}
                    alt={`指挥官：${commander}`}
                    onLoad={handleImageLoad}
                    onError={(e) => handleImageError(e, 'commanders', commander)}
                    error={renderErrorState(120)}
                    loader={<Skeleton animation image style={{ width: '100%', height: '120px' }} />}
                    actions={[
                      <div key="preview" style={{ color: colors.text.primary }}>点击预览</div>
                    ]}
                  />
                  <ImageLabel>指挥官：{commander}</ImageLabel>
                </ImageContainer>
              </Col>
            ))}
          </Row>

          <Row gutter={16} style={{ padding: '16px' }}>
            {mutations.map(mutation => (
              <Col span={8} key={mutation}>
                <ImageContainer>
                  <MutationImage 
                    src={getImagePath(mutation, 'mutations')}
                    alt={`突变因子：${mutation}`}
                    onLoad={handleImageLoad}
                    onError={(e) => handleImageError(e, 'mutations', mutation)}
                    error={renderErrorState(80)}
                    loader={<Skeleton animation image style={{ width: '100%', height: '80px' }} />}
                    actions={[
                      <div key="preview" style={{ color: colors.text.primary }}>点击预览</div>
                    ]}
                  />
                  <ImageLabel>突变因子：{mutation}</ImageLabel>
                </ImageContainer>
              </Col>
            ))}
          </Row>
        </ImagesCard>
      </RightColumn>
    </ResultLayout>
  );
};

export default MutationResult; 