import 'styled-components';
import { colors } from '../styles/theme';

declare module 'styled-components' {
  export interface DefaultTheme {
    colors: typeof colors;
  }
} 