**Potential fallouts**

1. Types of used lights (white, UV, florescent).

**Goals to calculate**

1. Horizontal dimention (Not needed)
2. Starting poistion \*
3. End position 
4. Track width \*
5. Distance b/w tracks \*
6. number of inbetween tracks \*
7. migration front \*
8. edge cut \*

**Pipeline**

1. clean the image.
2. calculate the goals.
3. detect tracks

**potential ways to solve**

1. Edge based lanes
   - sobel filter or canny filter.
   - Run vertical or horizontal projections, find peaks
   - check out this: https://vincmazet.github.io/bip/detection/edges.html#

2. Template based
   - check out Hough lines or RANSAC etc

3. Fourier-based lane-detection
   - check this out

4. Deep learning
   - Train/Fine-tune segmentation model (U-Net/ Mask-RCNN).
   - from mask get precise boundaries.

# 1. Edge based lanes

    - Sobel filter.
    - canny filter.

## 1.1 Sobel filter.

- link: https://www.youtube.com/watch?v=VL8PuOPjVjY

- Greyscale image --> apply filters (kernals) --> calculate magnitude --> Thresholding --> Binary mask

**Problems**

- tracks are good but we need a clean the image first properly and mark the tracks.
- different edges have diffferent thickness.

## 1.2 Canny filter.

### 1.3 HoughLinesP

- it is used to detect the straight line in the canny edge map (check this out)

# 4. Deep learning

## segmentation.

- is done using the SAM model (FASTSAM).

## marking

- for reference check this linke (https://gemini.google.com/app/e4fb5034267b2a40?_gl=1*rh7obz*_gcl_au*NzAwMDkzMjQuMTc2MTc0NjM1Nw.._ga*MTQxNDM5Mjk3OC4xNzYxNzQ2MzUy*_ga_WC57KJ50ZZ*czE3NjE3NDYzNTEkbzEkZzEkdDE3NjE3NDY2NjEkajU0JGwwJGgw)

### Connected grouping.

1. check the blobs along the y-axis and mark the start and end positions of each blob.
2. mark the centriod of each blob.
3. select the blob with the lowest width.
4. mark that width in y direction as a track.

# todo
+  do reprocessing from meneet.
+ check out if we can use clustering to mark the blobs properly. in segment.py
+ correct some images.

