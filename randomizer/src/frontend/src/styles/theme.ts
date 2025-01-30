/**
 * 主题样式定义
 */

import { ThemeConfig } from 'antd';
import { createGlobalStyle } from 'styled-components';

// 星际2主题色
export const colors = {
  primary: '#00A7E1', // 科技蓝
  secondary: '#6B4FBB', // 虚空紫
  accent: '#FF7F00', // 能量橙
  background: '#0A0A0A', // 深空黑
  surface: '#1A1A1A', // 面板黑
  border: '#2A2A2A', // 边框灰
  text: {
    primary: '#FFFFFF',
    secondary: '#AAAAAA',
    accent: '#00A7E1',
  },
  status: {
    success: '#4CAF50',
    warning: '#FF9800',
    error: '#F44336',
    info: '#2196F3',
  },
};

// Ant Design主题配置
export const antdTheme: ThemeConfig = {
  token: {
    colorPrimary: colors.primary,
    colorSuccess: colors.status.success,
    colorWarning: colors.status.warning,
    colorError: colors.status.error,
    colorInfo: colors.status.info,
    colorTextBase: colors.text.primary,
    colorBgBase: colors.background,
    borderRadius: 4,
    wireframe: false,
  },
  components: {
    Button: {
      primaryColor: colors.primary,
      defaultBg: colors.surface,
      defaultBorderColor: colors.border,
    },
    Card: {
      colorBgContainer: colors.surface,
    },
    Select: {
      colorBgContainer: colors.surface,
      colorBorder: colors.border,
    },
    Slider: {
      railBg: colors.border,
      trackBg: colors.primary,
      handleColor: colors.primary,
    },
  },
};

// 全局样式
export const GlobalStyle = createGlobalStyle`
  body {
    margin: 0;
    padding: 0;
    background-color: ${colors.background};
    color: ${colors.text.primary};
    font-family: "PingFang SC", "Microsoft YaHei", sans-serif;
  }

  // 星际2风格的标题
  h1, h2, h3, h4, h5, h6 {
    font-family: "Bank Gothic", "Orbitron", sans-serif;
    text-transform: uppercase;
    letter-spacing: 2px;
    color: ${colors.text.primary};
    text-shadow: 0 0 10px ${colors.primary}40;
  }

  // 科技感的分割线
  hr {
    border: none;
    height: 1px;
    background: linear-gradient(
      90deg,
      transparent,
      ${colors.primary}40,
      ${colors.primary},
      ${colors.primary}40,
      transparent
    );
  }

  // 自定义滚动条
  ::-webkit-scrollbar {
    width: 8px;
    height: 8px;
  }

  ::-webkit-scrollbar-track {
    background: ${colors.surface};
  }

  ::-webkit-scrollbar-thumb {
    background: ${colors.border};
    border-radius: 4px;
    
    &:hover {
      background: ${colors.primary}80;
    }
  }

  // 选中文本样式
  ::selection {
    background: ${colors.primary}40;
    color: ${colors.text.primary};
  }
`; 