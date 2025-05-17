import { Box, Heading, Text, Button, Container, VStack } from '@chakra-ui/react'
import { Link as RouterLink } from 'react-router-dom'

export default function NotFoundPage() {
  return (
    <Container centerContent py={20}>
      <VStack spacing={6} textAlign="center">
        <Heading as="h1" size="4xl">
          404
        </Heading>
        <Heading as="h2" size="xl">
          Page Not Found
        </Heading>
        <Text fontSize="lg">
          The page you're looking for doesn't exist or has been moved.
        </Text>
        <Button as={RouterLink} to="/" colorScheme="blue" size="lg">
          Return Home
        </Button>
      </VStack>
    </Container>
  )
}