import cv2
import numpy as np

#pixel coordinates from frame
px_points = np.array([
    [828, 468],
    [746, 578],
    [404, 439],
    [270, 539],
    [521, 351],
    [109, 659],
], dtype=np.float32)

#nba court coordinates (feet)
court_points = np.array([
    [17, 19],
    [33, 19],
    [17, 0],
    [33, 0],
    [0, 0],
    [47, 0],
], dtype=np.float32)

#computing homography
H, mask = cv2.findHomography(px_points, court_points, cv2.RANSAC, 5.0)

print("H matrix")
print(H)
print(f"\nInliers: {mask.sum()}/6")

#convert pixel points to estimated court coordinates
print("\nVerification — pixel → transformed court vs expected:")
print(f"{'Point':<8}{'Pixel':<20}{'Transformed':<25}{'Expected':<20}{'Error (ft)'}")
print("----")

for i, (px, ct) in enumerate(zip(px_points, court_points)):
    px_h = np.array([[px[0], px[1]]], dtype=np.float32).reshape(-1, 1, 2)
    transformed = cv2.perspectiveTransform(px_h, H)[0][0]
    #squared error
    error = np.sqrt((transformed[0] - ct[0])**2 + (transformed[1] - ct[1])**2)
    print(f"{i+1:<8}{str(tuple(px.astype(int))):<20}{str(tuple(transformed.round(2))):<25}{str(tuple(ct)):<20}{error:.3f}")

    #save for future reference
    # Save H for use in other scripts
np.save("models/homography_frame1.npy", H)
print("\nH matrix saved to models/homography_frame1.npy")
