/**
 * 应用入口组件
 */

import React from 'react';
import { ConfigProvider } from '@arco-design/web-react';
import { GlobalStyle } from './styles/theme';
import Home from './pages/Home';
import '@arco-design/web-react/dist/css/arco.css';

const App: React.FC = () => {
  return (
    <ConfigProvider>
      <GlobalStyle />
      <Home />
    </ConfigProvider>
  );
};

export default App; 