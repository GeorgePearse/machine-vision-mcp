import { useEffect, useState } from 'react'
import { Link as RouterLink } from 'react-router-dom'
import {
  Box,
  Heading,
  Text,
  Button,
  SimpleGrid,
  useColorModeValue,
  VStack,
  Image,
  Card,
  CardBody,
  Stack,
  Divider,
  CardFooter,
  ButtonGroup,
  Center,
  Container,
  Icon,
  HStack,
} from '@chakra-ui/react'
import { FaImage, FaSearchPlus, FaChartBar, FaRobot } from 'react-icons/fa'
import { listImages } from '../utils/api'

export default function HomePage() {
  const [images, setImages] = useState([])
  const [loading, setLoading] = useState(true)

  useEffect(() => {
    const fetchImages = async () => {
      try {
        const data = await listImages()
        setImages(data.images || [])
      } catch (error) {
        console.error('Error fetching images:', error)
      } finally {
        setLoading(false)
      }
    }

    fetchImages()
  }, [])

  return (
    <Container maxW="container.xl">
      <VStack spacing={8} as="section" textAlign="center" py={10}>
        <Heading
          as="h1"
          size="2xl"
          fontWeight="bold"
          color={useColorModeValue('gray.700', 'white')}
        >
          Machine Vision MCP
        </Heading>
        <Text fontSize="xl" color={useColorModeValue('gray.600', 'gray.300')} maxW="3xl">
          Connect LLMs to powerful scikit-image computer vision capabilities using
          a simple, intuitive interface
        </Text>
        <Button
          as={RouterLink}
          to="/upload"
          size="lg"
          colorScheme="blue"
          leftIcon={<FaImage />}
        >
          Upload an Image
        </Button>
      </VStack>

      <Box as="section" py={10}>
        <Heading as="h2" size="xl" mb={8} textAlign="center">
          Features
        </Heading>
        <SimpleGrid columns={{ base: 1, md: 3 }} spacing={10}>
          <FeatureCard
            icon={FaSearchPlus}
            title="Object Detection"
            description="Detect corners, blobs, and specific features in your images with powerful scikit-image algorithms."
          />
          <FeatureCard
            icon={FaChartBar}
            title="Measurement"
            description="Count pixels, measure areas, and analyze object properties with precision."
          />
          <FeatureCard
            icon={FaRobot}
            title="LLM Integration"
            description="Connect any LLM to access computer vision tools through a clean, consistent API."
          />
        </SimpleGrid>
      </Box>

      {images.length > 0 && (
        <Box as="section" py={10}>
          <Heading as="h2" size="xl" mb={8} textAlign="center">
            Recent Images
          </Heading>
          <SimpleGrid columns={{ base: 1, sm: 2, md: 3, lg: 4 }} spacing={6}>
            {images.slice(0, 8).map((image) => (
              <Card key={image.filename} maxW="sm" overflow="hidden">
                <CardBody>
                  <Image
                    src={image.file_path}
                    alt={image.filename}
                    borderRadius="lg"
                    height="200px"
                    width="100%"
                    objectFit="cover"
                  />
                  <Stack mt="6" spacing="3">
                    <Text fontSize="sm" color="gray.500">
                      {image.width} x {image.height} • {(image.size / 1024).toFixed(1)} KB
                    </Text>
                  </Stack>
                </CardBody>
                <Divider />
                <CardFooter>
                  <ButtonGroup spacing="2">
                    <Button
                      as={RouterLink}
                      to={`/analysis/${image.filename}`}
                      variant="solid"
                      colorScheme="blue"
                      size="sm"
                    >
                      Analyze
                    </Button>
                  </ButtonGroup>
                </CardFooter>
              </Card>
            ))}
          </SimpleGrid>
        </Box>
      )}
    </Container>
  )
}

function FeatureCard({ icon, title, description }) {
  return (
    <VStack
      bg={useColorModeValue('white', 'gray.800')}
      boxShadow="md"
      borderRadius="lg"
      p={6}
      spacing={4}
      align="center"
      borderWidth="1px"
      borderColor={useColorModeValue('gray.200', 'gray.700')}
    >
      <Center
        bg={useColorModeValue('blue.50', 'blue.900')}
        p={3}
        borderRadius="full"
        color={useColorModeValue('blue.600', 'blue.300')}
      >
        <Icon as={icon} boxSize={6} />
      </Center>
      <Heading as="h3" size="md">
        {title}
      </Heading>
      <Text textAlign="center" color={useColorModeValue('gray.600', 'gray.400')}>
        {description}
      </Text>
    </VStack>
  )
}