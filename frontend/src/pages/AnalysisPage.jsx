import { useState, useEffect, useRef } from 'react'
import { useParams, useNavigate } from 'react-router-dom'
import {
  Box,
  Button,
  Container,
  Flex,
  Grid,
  GridItem,
  Heading,
  Image,
  Select,
  Text,
  Textarea,
  VStack,
  HStack,
  useToast,
  Tabs,
  TabList,
  TabPanels,
  Tab,
  TabPanel,
  Card,
  CardBody,
  CardHeader,
  Badge,
  Spinner,
  Divider,
  useColorModeValue,
  FormControl,
  FormLabel,
} from '@chakra-ui/react'
import { analyzeImage } from '../utils/api'
import { FaRobot, FaImage, FaServer } from 'react-icons/fa'
import ChatInterface from '../components/ChatInterface'
import ResultVisualization from '../components/ResultVisualization'

// Analysis types
const ANALYSIS_TYPES = [
  { value: 'corner_detection', label: 'Corner Detection' },
  { value: 'blob_detection', label: 'Blob Detection' },
  { value: 'hog', label: 'Histogram of Oriented Gradients' },
  { value: 'daisy', label: 'DAISY Features' },
  { value: 'lbp', label: 'Local Binary Patterns' },
  { value: 'object_segmentation', label: 'Object Segmentation' },
  { value: 'pixel_count', label: 'Pixel Count' },
  { value: 'area_measurement', label: 'Area Measurement' },
]

export default function AnalysisPage() {
  const { id: imageFilename } = useParams()
  const navigate = useNavigate()
  const toast = useToast()
  
  const [image, setImage] = useState(null)
  const [loading, setLoading] = useState(false)
  const [analysisType, setAnalysisType] = useState('corner_detection')
  const [threshold, setThreshold] = useState(0.5)
  const [analysisResult, setAnalysisResult] = useState(null)
  const [parameters, setParameters] = useState({})
  const [userPrompt, setUserPrompt] = useState('')
  const [chatMessages, setChatMessages] = useState([])

  // Get image path
  const imagePath = imageFilename ? `/uploads/${imageFilename}` : null

  // Load image on mount
  useEffect(() => {
    if (!imageFilename) {
      navigate('/upload')
      return
    }

    // Load image info
    setImage({
      filename: imageFilename,
      path: imagePath,
    })
  }, [imageFilename, navigate, imagePath])

  // Handle analysis type change
  const handleAnalysisTypeChange = (e) => {
    setAnalysisType(e.target.value)
    setAnalysisResult(null)
  }

  // Handle threshold change
  const handleThresholdChange = (e) => {
    setThreshold(parseFloat(e.target.value))
  }

  // Analyze image using direct API call
  const handleDirectAnalysis = async () => {
    if (!image) return

    setLoading(true)
    try {
      const result = await analyzeImage({
        filename: image.filename,
        analysis_type: analysisType,
        threshold: threshold,
        parameters: parameters,
      })

      setAnalysisResult(result)

      toast({
        title: 'Analysis complete',
        description: result.message || 'Image analysis completed successfully',
        status: 'success',
        duration: 3000,
        isClosable: true,
      })
    } catch (error) {
      console.error('Analysis error:', error)
      toast({
        title: 'Analysis failed',
        description: error.response?.data?.detail || 'Failed to analyze image',
        status: 'error',
        duration: 5000,
        isClosable: true,
      })
    } finally {
      setLoading(false)
    }
  }

  // Handle sending prompt to Claude
  const handleSendPrompt = async () => {
    if (!userPrompt.trim()) return
    
    // Add user message to chat
    const userMessage = {
      role: 'user',
      content: userPrompt,
    }
    
    setChatMessages((prev) => [...prev, userMessage])
    
    // Create Claude assistant message
    setLoading(true)
    try {
      // Simulate Claude API call using our backend
      // In a real app, you would call Claude's API with the image and prompt
      
      // The prompt includes the image analysis request
      // Claude would call our analysis API and return the results
      
      // For this demo, we'll just call our API directly
      const result = await analyzeImage({
        filename: image.filename,
        analysis_type: analysisType,
        threshold: threshold,
        parameters: parameters,
      })
      
      setAnalysisResult(result)
      
      // Create a simulated Claude response
      const claudeMessage = {
        role: 'assistant',
        content: `I've analyzed the image using ${getAnalysisTypeLabel(analysisType)}. ${result.message}`,
        result: result,
      }
      
      setChatMessages((prev) => [...prev, claudeMessage])
    } catch (error) {
      console.error('Analysis error:', error)
      
      // Add error message to chat
      const errorMessage = {
        role: 'assistant',
        content: `I couldn't analyze the image. Error: ${error.response?.data?.detail || 'Unknown error'}`,
      }
      
      setChatMessages((prev) => [...prev, errorMessage])
    } finally {
      setLoading(false)
      setUserPrompt('')
    }
  }

  // Helper function to get label from analysis type
  const getAnalysisTypeLabel = (type) => {
    const analysisType = ANALYSIS_TYPES.find((item) => item.value === type)
    return analysisType ? analysisType.label : type
  }

  const bgColor = useColorModeValue('white', 'gray.800')
  const borderColor = useColorModeValue('gray.200', 'gray.700')

  return (
    <Container maxW="container.xl" py={6}>
      <Grid templateColumns={{ base: '1fr', lg: '1fr 1fr' }} gap={8}>
        {/* Image and Analysis Controls Panel */}
        <GridItem>
          <VStack spacing={6} align="stretch">
            <Heading as="h1" size="xl">
              Image Analysis
            </Heading>

            {/* Image Display */}
            {image && (
              <Box borderRadius="md" overflow="hidden" borderWidth="1px" borderColor={borderColor}>
                <Image 
                  src={image.path}
                  alt={image.filename}
                  maxH="400px"
                  w="100%"
                  objectFit="contain"
                />
              </Box>
            )}

            {/* Analysis Controls - Direct API Tab */}
            <Card variant="outline">
              <CardHeader pb={0}>
                <Heading size="md">Analysis Controls</Heading>
              </CardHeader>
              <CardBody>
                <Tabs variant="soft-rounded" colorScheme="blue">
                  <TabList mb={4}>
                    <Tab><HStack><FaServer size={14} /><Text>Direct API</Text></HStack></Tab>
                    <Tab><HStack><FaRobot size={14} /><Text>Claude API</Text></HStack></Tab>
                  </TabList>
                  <TabPanels>
                    {/* Direct API Tab */}
                    <TabPanel p={0}>
                      <VStack spacing={4} align="stretch">
                        <FormControl>
                          <FormLabel>Analysis Type</FormLabel>
                          <Select value={analysisType} onChange={handleAnalysisTypeChange}>
                            {ANALYSIS_TYPES.map((type) => (
                              <option key={type.value} value={type.value}>
                                {type.label}
                              </option>
                            ))}
                          </Select>
                        </FormControl>
                        
                        {(analysisType === 'object_segmentation' || analysisType === 'area_measurement') && (
                          <FormControl>
                            <FormLabel>Threshold (0-1)</FormLabel>
                            <HStack>
                              <input
                                type="range"
                                min="0"
                                max="1"
                                step="0.05"
                                value={threshold}
                                onChange={handleThresholdChange}
                                style={{ width: '100%' }}
                              />
                              <Text minW="40px" textAlign="right">{threshold.toFixed(2)}</Text>
                            </HStack>
                          </FormControl>
                        )}
                        
                        <Button
                          colorScheme="blue"
                          onClick={handleDirectAnalysis}
                          isLoading={loading}
                          leftIcon={<FaImage />}
                        >
                          Analyze Image
                        </Button>
                      </VStack>
                    </TabPanel>
                    
                    {/* Claude API Tab */}
                    <TabPanel p={0}>
                      <VStack spacing={4} align="stretch">
                        <Text>
                          Describe what you want to know about this image, and Claude will use 
                          scikit-image tools to analyze it.
                        </Text>
                        <Textarea
                          placeholder="Example: Count all the objects in this image and measure their areas."
                          value={userPrompt}
                          onChange={(e) => setUserPrompt(e.target.value)}
                          rows={4}
                        />
                        <Button
                          colorScheme="purple"
                          onClick={handleSendPrompt}
                          isLoading={loading}
                          leftIcon={<FaRobot />}
                        >
                          Ask Claude
                        </Button>
                      </VStack>
                    </TabPanel>
                  </TabPanels>
                </Tabs>
              </CardBody>
            </Card>
          </VStack>
        </GridItem>

        {/* Results Panel */}
        <GridItem>
          <VStack spacing={6} align="stretch">
            <Heading as="h2" size="xl">
              Results
            </Heading>

            {/* Claude Chat Interface */}
            {chatMessages.length > 0 && (
              <ChatInterface messages={chatMessages} />
            )}

            {/* Analysis Results */}
            {analysisResult && (
              <ResultVisualization result={analysisResult} analysisType={analysisType} />
            )}

            {/* Loading State */}
            {loading && (
              <Card>
                <CardBody>
                  <VStack spacing={4}>
                    <Spinner size="xl" color="blue.500" />
                    <Text>Analyzing image...</Text>
                  </VStack>
                </CardBody>
              </Card>
            )}

            {/* Empty State */}
            {!analysisResult && !loading && !chatMessages.length && (
              <Card variant="outline">
                <CardBody>
                  <VStack spacing={4} py={8}>
                    <Box p={3} borderRadius="full" bg="blue.50">
                      <FaImage size={28} color="#3182ce" />
                    </Box>
                    <Text color="gray.500" textAlign="center">
                      Select an analysis type and click "Analyze Image",
                      or ask Claude to analyze the image for you.
                    </Text>
                  </VStack>
                </CardBody>
              </Card>
            )}
          </VStack>
        </GridItem>
      </Grid>
    </Container>
  )
}