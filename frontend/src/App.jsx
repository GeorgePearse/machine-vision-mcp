import { Routes, Route } from 'react-router-dom'
import { Box } from '@chakra-ui/react'
import Header from './components/Header'
import Footer from './components/Footer'
import HomePage from './pages/HomePage'
import AnalysisPage from './pages/AnalysisPage'
import UploadPage from './pages/UploadPage'
import NotFoundPage from './pages/NotFoundPage'

function App() {
  return (
    <Box minH="100vh" display="flex" flexDirection="column">
      <Header />
      <Box flex="1" as="main" py={8} px={4}>
        <Routes>
          <Route path="/" element={<HomePage />} />
          <Route path="/upload" element={<UploadPage />} />
          <Route path="/analysis/:id" element={<AnalysisPage />} />
          <Route path="*" element={<NotFoundPage />} />
        </Routes>
      </Box>
      <Footer />
    </Box>
  )
}

export default App