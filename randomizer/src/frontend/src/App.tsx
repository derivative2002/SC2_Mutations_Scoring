/**
 * 应用入口组件
 */

import React from 'react';
import { ConfigProvider } from 'antd';
import { GlobalStyle, antdTheme } from './styles/theme';
import Home from './pages/Home';

const App: React.FC = () => {
  return (
    <ConfigProvider theme={antdTheme}>
      <GlobalStyle />
      <Home />
    </ConfigProvider>
  );
};

export default App; 