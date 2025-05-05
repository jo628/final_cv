import cv2
import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from tqdm import tqdm

class CustomSegmenter:
    def __init__(self, method='thresholding', threshold_value=127, n_clusters=3):
        """
        Initialize the custom segmenter.
        
        Args:
            method: Segmentation method ('thresholding', 'edge', 'region', 'clustering')
            threshold_value: Threshold value for thresholding method
            n_clusters: Number of clusters for clustering method
        """
        self.method = method
        self.threshold_value = threshold_value
        self.n_clusters = n_clusters
        
    def apply_thresholding(self, image, adaptive=True):
        """
        Apply thresholding segmentation.
        
        Args:
            image: Input image
            adaptive: Whether to use adaptive thresholding
            
        Returns:
            Segmented image
        """
        # Convert to grayscale if needed
        if len(image.shape) > 2:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
            
        if adaptive:
            # Adaptive thresholding
            return cv2.adaptiveThreshold(
                gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                cv2.THRESH_BINARY, 11, 2
            )
        else:
            # Global thresholding
            _, thresh = cv2.threshold(gray, self.threshold_value, 255, cv2.THRESH_BINARY)
            return thresh
    
    def apply_edge_based(self, image, low_threshold=50, high_threshold=150):
        """
        Apply edge-based segmentation using Canny edge detector and contour finding.
        
        Args:
            image: Input image
            low_threshold: Lower threshold for Canny
            high_threshold: Higher threshold for Canny
            
        Returns:
            Segmented image with detected objects
        """
        # Convert to grayscale if needed
        if len(image.shape) > 2:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
            
        # Apply Gaussian blur to reduce noise
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        
        # Detect edges using Canny
        edges = cv2.Canny(blurred, low_threshold, high_threshold)
        
        # Dilate to close gaps in edges
        kernel = np.ones((3, 3), np.uint8)
        dilated = cv2.dilate(edges, kernel, iterations=1)
        
        # Find contours
        contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Create a mask for the segmented regions
        mask = np.zeros_like(gray)
        
        # Filter contours by area and draw them
        min_area = 100  # Adjust based on your image size
        for contour in contours:
            if cv2.contourArea(contour) > min_area:
                cv2.drawContours(mask, [contour], -1, 255, -1)
        
        # Apply the mask to the original image
        if len(image.shape) > 2:
            segmented = np.zeros_like(image)
            segmented[mask == 255] = image[mask == 255]
        else:
            segmented = np.zeros_like(gray)
            segmented[mask == 255] = gray[mask == 255]
            
        return segmented
    
    def apply_region_based(self, image, seed_points=None):
        """
        Apply region-based segmentation using region growing from seeds.
        
        Args:
            image: Input image
            seed_points: List of seed points (x, y). If None, uses a grid of seeds.
            
        Returns:
            Segmented image
        """
        # Convert to grayscale if needed
        if len(image.shape) > 2:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
            
        # If no seed points, create a grid of seed points
        if seed_points is None:
            h, w = gray.shape
            step = 50  # Adjust based on your image size
            seed_points = [(x, y) for y in range(step, h, step) 
                         for x in range(step, w, step)]
        
        # Create a mask for the segmented regions
        mask = np.zeros_like(gray, dtype=np.uint8)
        
        # Set threshold for region growing
        threshold = 20  # Adjust based on your needs
        
        # For each seed point, perform region growing
        for seed_x, seed_y in seed_points:
            if 0 <= seed_x < gray.shape[1] and 0 <= seed_y < gray.shape[0]:
                if mask[seed_y, seed_x] == 0:  # Only grow from unvisited seeds
                    region_mask = self._region_grow(gray, seed_x, seed_y, threshold)
                    mask = np.maximum(mask, region_mask)
        
        # Apply the mask to the original image
        if len(image.shape) > 2:
            segmented = np.zeros_like(image)
            segmented[mask == 255] = image[mask == 255]
        else:
            segmented = np.zeros_like(gray)
            segmented[mask == 255] = gray[mask == 255]
            
        return segmented
    
    def _region_grow(self, image, seed_x, seed_y, threshold):
        """
        Helper method for region growing from a seed point.
        
        Args:
            image: Input grayscale image
            seed_x, seed_y: Seed point coordinates
            threshold: Intensity threshold for region growing
            
        Returns:
            Mask of the segmented region
        """
        h, w = image.shape
        seed_value = int(image[seed_y, seed_x])
        
        # Initialize mask and queue
        mask = np.zeros((h, w), dtype=np.uint8)
        queue = [(seed_x, seed_y)]
        visited = set(queue)
        
        # Define 4-connectivity neighbors
        neighbors = [(0, 1), (1, 0), (0, -1), (-1, 0)]
        
        # Region growing
        while queue:
            x, y = queue.pop(0)
            
            if abs(int(image[y, x]) - seed_value) <= threshold:
                mask[y, x] = 255
                
                # Add neighbors to queue
                for dx, dy in neighbors:
                    nx, ny = x + dx, y + dy
                    if (0 <= nx < w and 0 <= ny < h and 
                        (nx, ny) not in visited):
                        queue.append((nx, ny))
                        visited.add((nx, ny))
        
        return mask
    
    def apply_clustering(self, image):
        """
        Apply clustering-based segmentation using K-means.
        
        Args:
            image: Input image
            
        Returns:
            Segmented image
        """
        # Reshape the image for clustering
        pixels = image.reshape((-1, 3)).astype(np.float32)
        
        # Apply K-means clustering with lower iterations for CPU
        kmeans = KMeans(n_clusters=self.n_clusters, random_state=42, n_init=10, max_iter=100)
        labels = kmeans.fit_predict(pixels)
        
        # Map labels back to image shape
        segmented = labels.reshape(image.shape[:2])
        
        # Normalize to 0-255 range
        segmented = (segmented * (255 // (self.n_clusters - 1))).astype(np.uint8)
        
        return segmented
    
    def segment(self, image):
        """
        Apply the selected segmentation method to an image.
        
        Args:
            image: Input image (BGR format)
            
        Returns:
            Segmented image
        """
        # Copy image to avoid modifying the original
        img = image.copy()
        
        # Make sure pixel values are in 0-255 range
        if img.dtype == np.float32 or img.dtype == np.float64:
            img = (img * 255).astype(np.uint8)
        
        # Apply the selected segmentation method
        if self.method == 'thresholding':
            return self.apply_thresholding(img)
        elif self.method == 'edge':
            return self.apply_edge_based(img)
        elif self.method == 'region':
            return self.apply_region_based(img)
        elif self.method == 'clustering':
            return self.apply_clustering(img)
        else:
            raise ValueError(f"Unknown segmentation method: {self.method}")
    
    def batch_segment(self, images, with_tqdm=True):
        """
        Segment a batch of images.
        
        Args:
            images: List of input images
            with_tqdm: Whether to use tqdm for progress tracking
            
        Returns:
            List of segmented images
        """
        if with_tqdm:
            iterator = tqdm(images, desc="Segmenting images")
        else:
            iterator = images
            
        return [self.segment(img) for img in iterator]
    
    def visualize_segmentation(self, original, segmented, figsize=(12, 6)):
        """
        Visualize the original and segmented images side by side.
        
        Args:
            original: Original image
            segmented: Segmented image
            figsize: Figure size
        """
        plt.figure(figsize=figsize)
        
        plt.subplot(1, 2, 1)
        if len(original.shape) == 3:
            plt.imshow(cv2.cvtColor(original, cv2.COLOR_BGR2RGB))
        else:
            plt.imshow(original, cmap='gray')
        plt.title('Original Image')
        plt.axis('off')
        
        plt.subplot(1, 2, 2)
        if len(segmented.shape) == 3:
            plt.imshow(cv2.cvtColor(segmented, cv2.COLOR_BGR2RGB))
        else:
            plt.imshow(segmented, cmap='gray')
        plt.title('Segmented Image')
        plt.axis('off')
        
        plt.tight_layout()
        plt.show()