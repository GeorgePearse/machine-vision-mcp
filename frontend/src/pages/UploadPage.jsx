import { useState, useCallback } from 'react'
import { useNavigate } from 'react-router-dom'
import {
  Box,
  Button,
  Container,
  Heading,
  Text,
  VStack,
  useToast,
  Image,
  Stack,
  Flex,
  useColorModeValue,
  Icon,
  Progress,
} from '@chakra-ui/react'
import { useDropzone } from 'react-dropzone'
import { FaFileUpload, FaImage } from 'react-icons/fa'
import { uploadImage } from '../utils/api'

export default function UploadPage() {
  const [file, setFile] = useState(null)
  const [preview, setPreview] = useState(null)
  const [uploading, setUploading] = useState(false)
  const [uploadProgress, setUploadProgress] = useState(0)
  const toast = useToast()
  const navigate = useNavigate()

  const onDrop = useCallback((acceptedFiles) => {
    const selectedFile = acceptedFiles[0]
    if (selectedFile) {
      setFile(selectedFile)
      
      // Create preview URL
      const previewUrl = URL.createObjectURL(selectedFile)
      setPreview(previewUrl)
    }
  }, [])

  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    onDrop,
    accept: {
      'image/*': ['.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tiff']
    },
    maxFiles: 1,
    maxSize: 10 * 1024 * 1024, // 10MB
  })

  const handleUpload = async () => {
    if (!file) {
      toast({
        title: 'No file selected',
        description: 'Please select an image to upload.',
        status: 'warning',
        duration: 3000,
        isClosable: true,
      })
      return
    }

    setUploading(true)
    setUploadProgress(0)
    
    // Simulate progress
    const progressInterval = setInterval(() => {
      setUploadProgress((prev) => {
        if (prev >= 90) {
          clearInterval(progressInterval)
          return prev
        }
        return prev + 10
      })
    }, 300)

    try {
      const uploadedImage = await uploadImage(file)
      clearInterval(progressInterval)
      setUploadProgress(100)
      
      toast({
        title: 'Upload successful',
        description: 'Your image has been uploaded successfully.',
        status: 'success',
        duration: 3000,
        isClosable: true,
      })
      
      // Navigate to analysis page
      navigate(`/analysis/${uploadedImage.filename}`)
    } catch (error) {
      clearInterval(progressInterval)
      console.error('Upload error:', error)
      
      toast({
        title: 'Upload failed',
        description: error.response?.data?.detail || 'Failed to upload image.',
        status: 'error',
        duration: 5000,
        isClosable: true,
      })
    } finally {
      setUploading(false)
    }
  }

  const bgColor = useColorModeValue('gray.50', 'gray.700')
  const borderColor = useColorModeValue('gray.200', 'gray.600')
  const activeBorderColor = useColorModeValue('blue.400', 'blue.300')

  return (
    <Container maxW="container.md" py={10}>
      <VStack spacing={8} align="stretch">
        <Box textAlign="center">
          <Heading as="h1" size="xl" mb={4}>
            Upload an Image
          </Heading>
          <Text color={useColorModeValue('gray.600', 'gray.400')}>
            Upload an image to analyze with scikit-image computer vision tools
          </Text>
        </Box>

        <Box
          {...getRootProps()}
          p={6}
          borderWidth={2}
          borderRadius="md"
          borderStyle="dashed"
          borderColor={isDragActive ? activeBorderColor : borderColor}
          bg={bgColor}
          cursor="pointer"
          transition="all 0.2s"
          _hover={{
            borderColor: activeBorderColor,
          }}
        >
          <input {...getInputProps()} />
          <VStack spacing={4} py={preview ? 0 : 10}>
            {!preview ? (
              <>
                <Icon as={FaFileUpload} boxSize={12} color="gray.400" />
                <Text textAlign="center">
                  {isDragActive
                    ? "Drop the image here"
                    : "Drag and drop an image here, or click to select"}
                </Text>
                <Text fontSize="sm" color="gray.500">
                  Supported formats: JPG, PNG, GIF, BMP, TIFF (max 10MB)
                </Text>
              </>
            ) : (
              <Box py={4}>
                <Image
                  src={preview}
                  alt="Preview"
                  maxH="300px"
                  mx="auto"
                  borderRadius="md"
                  objectFit="contain"
                />
                <Text mt={2} fontSize="sm" color="gray.500" textAlign="center">
                  {file.name} ({(file.size / 1024).toFixed(1)} KB)
                </Text>
              </Box>
            )}
          </VStack>
        </Box>

        {uploading && (
          <Box>
            <Text mb={2}>Uploading image...</Text>
            <Progress value={uploadProgress} size="sm" colorScheme="blue" borderRadius="md" />
          </Box>
        )}

        <Flex justifyContent="center">
          <Button
            colorScheme="blue"
            size="lg"
            isLoading={uploading}
            onClick={handleUpload}
            loadingText="Uploading"
            disabled={!file || uploading}
            leftIcon={<FaImage />}
          >
            Upload and Analyze
          </Button>
        </Flex>
      </VStack>
    </Container>
  )
}