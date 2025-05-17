import { useRef, useEffect } from 'react'
import {
  Box,
  VStack,
  HStack,
  Text,
  Avatar,
  Flex,
  useColorModeValue,
  Divider,
} from '@chakra-ui/react'
import ResultVisualization from './ResultVisualization'

export default function ChatInterface({ messages }) {
  const chatEndRef = useRef(null)

  // Auto-scroll to bottom on new messages
  useEffect(() => {
    chatEndRef.current?.scrollIntoView({ behavior: 'smooth' })
  }, [messages])

  const userBgColor = useColorModeValue('blue.50', 'blue.900')
  const assistantBgColor = useColorModeValue('gray.50', 'gray.700')
  const borderColor = useColorModeValue('gray.200', 'gray.600')

  return (
    <Box 
      borderWidth="1px" 
      borderColor={borderColor} 
      borderRadius="md" 
      overflow="hidden"
      maxH="600px"
      overflowY="auto"
    >
      <VStack spacing={0} align="stretch" divider={<Divider />}>
        {messages.map((message, index) => (
          <Box 
            key={index} 
            bg={message.role === 'user' ? userBgColor : assistantBgColor}
            p={4}
          >
            <HStack spacing={3} align="flex-start">
              <Avatar 
                size="sm" 
                name={message.role === 'user' ? 'User' : 'Claude'}
                src={message.role === 'assistant' ? '/favicon.svg' : undefined}
                bg={message.role === 'user' ? 'blue.500' : 'purple.500'}
              />
              <VStack spacing={3} align="stretch" flex={1}>
                <Text fontSize="sm" fontWeight="bold">
                  {message.role === 'user' ? 'You' : 'Claude'}
                </Text>
                <Text>{message.content}</Text>
                
                {/* Display analysis result if exists */}
                {message.role === 'assistant' && message.result && (
                  <Box mt={3}>
                    <ResultVisualization result={message.result} />
                  </Box>
                )}
              </VStack>
            </HStack>
          </Box>
        ))}
        <div ref={chatEndRef} />
      </VStack>
    </Box>
  )
}