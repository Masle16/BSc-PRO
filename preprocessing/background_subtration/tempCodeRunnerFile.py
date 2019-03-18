    # Remove noise
    kernel = np.ones((2, 6), np.uint8)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)