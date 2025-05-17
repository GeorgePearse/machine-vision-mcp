FROM node:18-alpine

WORKDIR /app

# Copy package.json and package-lock.json
COPY package*.json ./

# Install dependencies
RUN npm install

# Copy the rest of the frontend code
COPY . .

# Expose the port
EXPOSE 5173

# Start the app
CMD ["npm", "run", "dev", "--", "--host", "0.0.0.0"]