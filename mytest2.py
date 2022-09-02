# Open and Convert the input image from BGR to GRAYSCALE,Image1 is Template image and Image2 is Search image
MIN_MATCH_COUNT = 3
image1 = cv.imread(filename = 'Figures/image1.png',
                   flags = cv.IMREAD_GRAYSCALE)

# Open and Convert the training-set image from BGR to GRAYSCALE
image2 = cv.imread(filename = 'Figures/image2.jpg',
                   flags = cv.IMREAD_GRAYSCALE)
img_rgb = cv.imread(filename = 'Figures/image1.png')

# Could not open or find the images
if image1 is None or image2 is None:
    print('\nCould not open or find the images.')
    exit(0)

# Find the keypoints and compute
# the descriptors for input image
globals.keypoints1, globals.descriptors1 = features.features(image1)

# Print
print('\nInput image:\n')

# Print infos for input image
features.prints(keypoints = globals.keypoints1,
                descriptor = globals.descriptors1)

# Find the keypoints and compute
# the descriptors for training-set image
globals.keypoints2, globals.descriptors2 = features.features(image2)

x = np.array([globals.keypoints2[0].pt])

for i in range(len(globals.keypoints2)):
    x = np.append(x, [globals.keypoints2[i].pt], axis=0)

x = x[1:len(x)]

bandwidth = estimate_bandwidth(x, quantile=0.1, n_samples=500)

# Print
print('Training-set image:\n')

# Print infos for training-set image
features.prints(keypoints = globals.keypoints2,
                descriptor = globals.descriptors2)
# finding clusters in search image using mean shift algorithm
ms = MeanShift(bandwidth=bandwidth, bin_seeding=True, cluster_all=True)
ms.fit(x)
labels = ms.labels_
cluster_centers = ms.cluster_centers_

labels_unique = np.unique(labels)
n_clusters_ = len(labels_unique)
print("number of estimated clusters : %d" % n_clusters_)

s = [None] * n_clusters_
for i in range(n_clusters_):
    l = ms.labels_
    d, = np.where(l == i)
    # print(d.__len__())
    s[i] = list(globals.keypoints2[xx] for xx in d)

des2_ = globals.descriptors2
# for every cluster, apply FLANN feature matching
for i in range(n_clusters_):

    kp2 = s[i]
    l = ms.labels_
    d, = np.where(l == i)
    des2 = des2_[d,]

    if (len(kp2) < 2 or len(kp1) < 2):
        continue
# Matcher
    output = features.matcher(image1 = image1,
                          image2 = image2,
                          keypoints1 = globals.keypoints1,
                          keypoints2 = globals.keypoints2,
                          descriptors1 = globals.descriptors1,
                          descriptors2 = globals.descriptors2,
                          matcher = arguments.matcher,
                          descriptor = arguments.descriptor)
    if len(output[4]) > 3:
        src_pts = np.float32([output[2][m.queryIdx].pt for m in output[4]]).reshape(-1, 1, 2)
        dst_pts = np.float32([output[3][m.trainIdx].pt for m in output[4]]).reshape(-1, 1, 2)

        M, mask = cv.findHomography(src_pts, dst_pts, cv.RANSAC, 2)

        if M is None:
            print("No Homography")
        else:
            h, w = image1.shape
        corners = np.float32([[0, 0], [0, h - 1], [w - 1, h - 1], [w - 1, 0]]).reshape(-1, 1, 2)
        transformedCorners = cv.perspectiveTransform(corners, M)

        x = int(transformedCorners[0][0][0])
        y = int(transformedCorners[0][0][1])
        # print(transformedCorners)
        # print(x)
        # print(y)
        cv.rectangle(img_rgb, (x, y), (x + w, y + h), (0, 0, 255), 3)

        # Draw a polygon on the second image joining the transformed corners
        image2 = cv.polylines(image2, [np.int32(transformedCorners)], True, (0, 0, 255), 2, cv.LINE_AA)

    else:
        print("Not enough matches are found - %d/%d" % (len(output[4]), MIN_MATCH_COUNT))
        matchesMask = None
#save result
save_figure.saveResult((output = image2,
			             matcher = arguments.matcher,
                         descriptor = arguments.descriptor))
# Save Figure Matcher
save_figures.saveMatcher(output = output,
			             matcher = arguments.matcher,
                         descriptor = arguments.descriptor)

# Save keypoints and descriptors into a file
# from input image
outputs.saveKeypointsAndDescriptors(keypoints = globals.keypoints1,
								    descriptors = globals.descriptors1,
                                    matcher = arguments.matcher,
                                    descriptor = arguments.descriptor,
                                    flags = 1)

# Save keypoints and descriptors into a file
# from training-set image
outputs.saveKeypointsAndDescriptors(keypoints = globals.keypoints2,
								    descriptors = globals.descriptors2,
                                    matcher = arguments.matcher,
                                    descriptor = arguments.descriptor,
                                    flags = 2)

# Print
print('Done!\n')

# Print
print('Feature Description and Matching executed with success!')