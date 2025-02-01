/**
 * 主题样式定义
 */

import { createGlobalStyle } from 'styled-components';

// 深色主题的颜色
export const colors = {
  primary: '#165DFF',  // 主题蓝
  secondary: '#722ED1', // 紫色
  accent: '#FF7D00',   // 橙色
  background: '#17171A', // 深色背景
  surface: '#232324',   // 深色表面
  surfaceLight: '#2A2A2B', // 稍亮的表面色
  border: '#363637',   // 深色边框
  text: {
    primary: '#FFFFFF',   // 主要文字
    secondary: '#86868A', // 次要文字
    accent: '#4080FF',    // 强调文字
  },
  status: {
    success: '#00B42A', // 成功绿
    warning: '#FF7D00', // 警告橙
    error: '#F53F3F',   // 错误红
    info: '#165DFF',    // 信息蓝
  },
};

// 全局样式
export const GlobalStyle = createGlobalStyle`
  body {
    margin: 0;
    padding: 0;
    background-color: ${colors.background};
    color: ${colors.text.primary};
    font-family: "PingFang SC", "Microsoft YaHei", -apple-system, BlinkMacSystemFont, "Segoe UI", 
                 "Roboto", "Oxygen", "Ubuntu", "Cantarell", "Fira Sans", "Droid Sans", 
                 "Helvetica Neue", sans-serif;
  }

  // 标题样式
  h1, h2, h3, h4, h5, h6 {
    font-family: "PingFang SC", "Microsoft YaHei", sans-serif;
    color: ${colors.text.primary};
    font-weight: 600;
  }

  // 分割线
  hr {
    border: none;
    height: 1px;
    background: ${colors.border};
  }

  // 自定义滚动条
  ::-webkit-scrollbar {
    width: 8px;
    height: 8px;
  }

  ::-webkit-scrollbar-track {
    background: ${colors.background};
  }

  ::-webkit-scrollbar-thumb {
    background: ${colors.border};
    border-radius: 4px;
    
    &:hover {
      background: ${colors.primary}40;
    }
  }

  // 选中文本样式
  ::selection {
    background: ${colors.primary}20;
    color: ${colors.text.primary};
  }

  // Arco Design 全局样式覆盖
  .arco-btn {
    border-radius: 4px;
  }

  .arco-card {
    border-radius: 8px;
    background: ${colors.surface};
    border-color: ${colors.border};
    
    .arco-card-header {
      border-bottom-color: ${colors.border};
    }
  }

  .arco-tag {
    border-radius: 4px;
  }

  .arco-slider {
    .arco-slider-road {
      height: 4px;
      background: ${colors.border};
    }
    .arco-slider-track {
      background: ${colors.primary};
      height: 4px;
    }
    .arco-slider-button {
      width: 16px;
      height: 16px;
      border-color: ${colors.primary};
      margin-top: -6px;  // 调整按钮垂直位置
    }
    .arco-slider-marks-text {
      margin-top: 8px;  // 调整刻度文字的位置
      color: ${colors.text.secondary};
    }
    .arco-slider-dot {
      width: 8px;
      height: 8px;
      margin-top: -2px;  // 调整刻度点的垂直位置
      border-color: ${colors.border};
      
      &.arco-slider-dot-active {
        border-color: ${colors.primary};
      }
    }
  }

  .arco-space {
    width: 100%;
  }
`; 