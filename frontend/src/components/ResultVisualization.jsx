import {
  Box,
  VStack,
  Image,
  Text,
  Heading,
  Grid,
  GridItem,
  Stat,
  StatLabel,
  StatNumber,
  StatHelpText,
  Table,
  Thead,
  Tbody,
  Tr,
  Th,
  Td,
  Badge,
  Card,
  CardBody,
  CardHeader,
  SimpleGrid,
  HStack,
  Tag,
  Divider,
  useColorModeValue,
} from '@chakra-ui/react'

export default function ResultVisualization({ result, analysisType }) {
  // Handle null or undefined result
  if (!result) return null

  const borderColor = useColorModeValue('gray.200', 'gray.600')
  const cardBg = useColorModeValue('white', 'gray.800')

  // Get visualization image
  const visualizationImage = result.visualization || null
  const visualizationSrc = visualizationImage ? `data:image/png;base64,${visualizationImage}` : null

  return (
    <Card variant="outline">
      <CardHeader pb={2}>
        <Heading size="md">Analysis Results</Heading>
        {result.message && (
          <Text mt={2} color="gray.600">
            {result.message}
          </Text>
        )}
      </CardHeader>
      <Divider />
      <CardBody>
        <VStack spacing={6} align="stretch">
          {/* Visualization Image */}
          {visualizationSrc && (
            <Box borderWidth="1px" borderColor={borderColor} borderRadius="md" overflow="hidden">
              <Image
                src={visualizationSrc}
                alt="Visualization"
                maxH="400px"
                w="100%"
                objectFit="contain"
              />
            </Box>
          )}

          {/* Different result displays based on type */}
          
          {/* Corner Detection */}
          {result.corners && (
            <Box>
              <Stat mb={4}>
                <StatLabel>Corners Detected</StatLabel>
                <StatNumber>{result.corners.length}</StatNumber>
              </Stat>
              
              {result.corners.length > 0 && (
                <Box maxH="200px" overflowY="auto" borderWidth="1px" borderColor={borderColor} borderRadius="md">
                  <Table size="sm" variant="simple">
                    <Thead>
                      <Tr>
                        <Th>Y</Th>
                        <Th>X</Th>
                        <Th>Score</Th>
                      </Tr>
                    </Thead>
                    <Tbody>
                      {result.corners.map((corner, index) => (
                        <Tr key={index}>
                          <Td>{corner[0]}</Td>
                          <Td>{corner[1]}</Td>
                          <Td>{result.scores ? result.scores[index].toFixed(4) : 'N/A'}</Td>
                        </Tr>
                      ))}
                    </Tbody>
                  </Table>
                </Box>
              )}
            </Box>
          )}
          
          {/* Blob Detection */}
          {result.blobs && (
            <Box>
              <Stat mb={4}>
                <StatLabel>Blobs Detected</StatLabel>
                <StatNumber>{result.blobs.length}</StatNumber>
              </Stat>
              
              {result.blobs.length > 0 && (
                <Box maxH="200px" overflowY="auto" borderWidth="1px" borderColor={borderColor} borderRadius="md">
                  <Table size="sm" variant="simple">
                    <Thead>
                      <Tr>
                        <Th>Y</Th>
                        <Th>X</Th>
                        <Th>Size</Th>
                      </Tr>
                    </Thead>
                    <Tbody>
                      {result.blobs.map((blob, index) => (
                        <Tr key={index}>
                          <Td>{blob[0].toFixed(1)}</Td>
                          <Td>{blob[1].toFixed(1)}</Td>
                          <Td>{blob[2].toFixed(1)}</Td>
                        </Tr>
                      ))}
                    </Tbody>
                  </Table>
                </Box>
              )}
            </Box>
          )}
          
          {/* Template Matching */}
          {result.matches && (
            <Box>
              <Stat mb={4}>
                <StatLabel>Matches Found</StatLabel>
                <StatNumber>{result.matches.length}</StatNumber>
              </Stat>
              
              <Heading size="sm" mb={2}>Best Match</Heading>
              <HStack mb={4}>
                <Tag colorScheme="green">Position: ({result.best_match[0]}, {result.best_match[1]})</Tag>
                <Tag colorScheme="blue">Score: {result.best_score.toFixed(4)}</Tag>
              </HStack>
              
              {result.matches.length > 0 && (
                <Box maxH="200px" overflowY="auto" borderWidth="1px" borderColor={borderColor} borderRadius="md">
                  <Table size="sm" variant="simple">
                    <Thead>
                      <Tr>
                        <Th>Y</Th>
                        <Th>X</Th>
                        <Th>Score</Th>
                      </Tr>
                    </Thead>
                    <Tbody>
                      {result.matches.map((match, index) => (
                        <Tr key={index}>
                          <Td>{match[0]}</Td>
                          <Td>{match[1]}</Td>
                          <Td>{result.scores ? result.scores[index].toFixed(4) : 'N/A'}</Td>
                        </Tr>
                      ))}
                    </Tbody>
                  </Table>
                </Box>
              )}
            </Box>
          )}
          
          {/* Object Segmentation */}
          {result.object_count !== undefined && (
            <SimpleGrid columns={{ base: 1, md: 2 }} spacing={4}>
              <Stat>
                <StatLabel>Objects Detected</StatLabel>
                <StatNumber>{result.object_count}</StatNumber>
              </Stat>
              
              {result.areas && (
                <Stat>
                  <StatLabel>Total Area</StatLabel>
                  <StatNumber>
                    {result.areas.reduce((sum, area) => sum + area, 0).toFixed(0)}
                  </StatNumber>
                  <StatHelpText>pixels</StatHelpText>
                </Stat>
              )}
            </SimpleGrid>
          )}
          
          {/* Pixel Count */}
          {result.counted_pixels !== undefined && (
            <SimpleGrid columns={{ base: 1, md: 2 }} spacing={4}>
              <Stat>
                <StatLabel>Counted Pixels</StatLabel>
                <StatNumber>{result.counted_pixels.toLocaleString()}</StatNumber>
                <StatHelpText>
                  {result.percentage.toFixed(2)}% of image
                </StatHelpText>
              </Stat>
              
              <Stat>
                <StatLabel>Total Pixels</StatLabel>
                <StatNumber>{result.total_pixels.toLocaleString()}</StatNumber>
              </Stat>
            </SimpleGrid>
          )}
          
          {/* Area Measurement */}
          {result.object_area_pixels !== undefined && (
            <SimpleGrid columns={{ base: 1, md: 2 }} spacing={4}>
              <Stat>
                <StatLabel>Object Area</StatLabel>
                <StatNumber>{result.object_area_pixels.toLocaleString()}</StatNumber>
                <StatHelpText>pixels</StatHelpText>
              </Stat>
              
              {result.object_area_units !== undefined && (
                <Stat>
                  <StatLabel>Object Area (Units)</StatLabel>
                  <StatNumber>{result.object_area_units.toFixed(2)}</StatNumber>
                  <StatHelpText>{result.unit}²</StatHelpText>
                </Stat>
              )}
            </SimpleGrid>
          )}
          
          {/* Feature extraction results (HOG, DAISY, etc.) */}
          {result.feature_length && (
            <Stat>
              <StatLabel>Feature Vector Length</StatLabel>
              <StatNumber>{result.feature_length}</StatNumber>
            </Stat>
          )}
          
          {result.keypoint_count && (
            <Stat>
              <StatLabel>Keypoints Detected</StatLabel>
              <StatNumber>{result.keypoint_count}</StatNumber>
            </Stat>
          )}
          
          {result.histogram_bins && (
            <Stat>
              <StatLabel>Histogram Bins</StatLabel>
              <StatNumber>{result.histogram_bins}</StatNumber>
            </Stat>
          )}
        </VStack>
      </CardBody>
    </Card>
  )
}