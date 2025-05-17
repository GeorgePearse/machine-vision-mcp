import { extendTheme } from '@chakra-ui/react'

const config = {
  initialColorMode: 'light',
  useSystemColorMode: false,
}

const colors = {
  brand: {
    50: '#e0f7ff',
    100: '#b9e7ff',
    200: '#8dd5ff',
    300: '#5ec3ff',
    400: '#36b2ff',
    500: '#0099ff',
    600: '#0078cc',
    700: '#005999',
    800: '#003a66',
    900: '#001c33',
  },
}

const theme = extendTheme({ config, colors })

export default theme