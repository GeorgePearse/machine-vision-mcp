import {
  Box,
  Container,
  Stack,
  Text,
  Link,
  useColorModeValue,
} from '@chakra-ui/react'
import { ExternalLinkIcon } from '@chakra-ui/icons'

export default function Footer() {
  return (
    <Box
      bg={useColorModeValue('gray.50', 'gray.900')}
      color={useColorModeValue('gray.700', 'gray.200')}
      mt={8}
    >
      <Container
        as={Stack}
        maxW={'6xl'}
        py={4}
        direction={{ base: 'column', md: 'row' }}
        spacing={4}
        justify={{ base: 'center', md: 'space-between' }}
        align={{ base: 'center', md: 'center' }}
      >
        <Text>© {new Date().getFullYear()} Machine Vision MCP. All rights reserved</Text>
        <Stack direction={'row'} spacing={6}>
          <Link href="https://scikit-image.org/" isExternal>
            scikit-image <ExternalLinkIcon mx="2px" />
          </Link>
          <Link href="https://github.com/jlowin/fastmcp" isExternal>
            fast-mcp <ExternalLinkIcon mx="2px" />
          </Link>
        </Stack>
      </Container>
    </Box>
  )
}