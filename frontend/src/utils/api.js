import axios from 'axios'

const API_URL = '/api'

// Create axios instance
const api = axios.create({
  baseURL: API_URL,
  headers: {
    'Content-Type': 'application/json',
  },
})

// API functions for images
export const uploadImage = async (file) => {
  const formData = new FormData()
  formData.append('file', file)
  
  const response = await api.post('/upload', formData, {
    headers: {
      'Content-Type': 'multipart/form-data',
    },
  })
  
  return response.data
}

export const listImages = async () => {
  const response = await api.get('/images')
  return response.data
}

export const deleteImage = async (filename) => {
  const response = await api.delete(`/images/${filename}`)
  return response.data
}

// API functions for analysis
export const analyzeImage = async (data) => {
  const response = await api.post('/analyze', data)
  return response.data
}

export const analyzePixels = async (data) => {
  const response = await api.post('/analyze/pixels', data)
  return response.data
}

// Health check
export const healthCheck = async () => {
  const response = await api.get('/health')
  return response.data
}

export default api