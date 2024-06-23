# import cv2
# import imutils
# import numpy as np
# import easyocr
# import os

# def process_image(filepath):
#     try:
#         image_path = filepath
#         img = cv2.imread(image_path)
#         gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

#         proses1 = gray

#         bfilter = cv2.bilateralFilter(gray, 11, 17, 17)  # Noise reduction

#         proses2 = bfilter

#         edged = cv2.Canny(bfilter, 30, 200)  # Edge detection

#         proses3 = edged

#         # Create folder if it does not exist
#         edged_folder = 'edged_images'
#         os.makedirs(edged_folder, exist_ok=True)

#         # Save edged image
#         edged_image_path = os.path.join(edged_folder, os.path.basename(filepath))
#         cv2.imwrite(edged_image_path, edged)
#         print(f'Edged image saved to: {edged_image_path}')

#         keypoints = cv2.findContours(edged.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
#         contours = imutils.grab_contours(keypoints)
#         contours = sorted(contours, key=cv2.contourArea, reverse=True)[:10]

#         location = None
#         for contour in contours:
#             approx = cv2.approxPolyDP(contour, 10, True)
#             if len(approx) == 4:
#                 location = approx
#                 break

#         if location is None:
#             raise ValueError("No valid contours found for OCR")

#         mask = np.zeros(gray.shape, np.uint8)
#         new_image = cv2.drawContours(mask, [location], 0, 255, -1)
#         new_image = cv2.bitwise_and(img, img, mask=mask)

#         (x, y) = np.where(mask == 255)
#         (x1, y1) = (np.min(x), np.min(y))
#         (x2, y2) = (np.max(x), np.max(y))

#         cropped_image = gray[x1:x2+1, y1:y2+1]

#         # Print cropped image dimensions for debugging
#         print("Cropped image shape:", cropped_image.shape)

#         # Create folder if it does not exist
#         output_folder = 'cropped_images'
#         os.makedirs(output_folder, exist_ok=True)

#         # Save cropped image
#         cropped_image_path = os.path.join(output_folder, os.path.basename(filepath))
#         cv2.imwrite(cropped_image_path, cropped_image)
#         print(f'Cropped image saved to: {cropped_image_path}')

#         proses4 = cropped_image

#         # Initialize EasyOCR reader
#         reader = easyocr.Reader(['en'])

#         # Perform OCR on cropped image
#         result = reader.readtext(cropped_image)



#         if result:
#             text = result[0][-2]
#             print(text)
#             return text
#         else:
#             return "No text found"

#     except Exception as e:
#         print("Error during image processing or OCR:", e)
#         return "No text found"



import cv2
import imutils
import numpy as np
import easyocr
import os

def process_image(filepath):
    try:
        # ============================PROSES 1==========================================
        image_path = filepath
        img = cv2.imread(image_path)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        output_dir = os.path.join('static', 'gray_images')
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        gray_image_path = os.path.join(output_dir, os.path.basename(image_path))
        cv2.imwrite(gray_image_path, gray)
        proses1 = gray_image_path
        print(f'Edged image saved to: {gray_image_path}')
        # ======================================================================


        # ===========================PROSES 2===========================================
        bfilter = cv2.bilateralFilter(gray, 11, 17, 17)  # Noise reduction
        filtered_output_dir = os.path.join('static', 'filtered_image')
        if not os.path.exists(filtered_output_dir):
            os.makedirs(filtered_output_dir)
        filtered_image_path = os.path.join(filtered_output_dir, os.path.basename(image_path))
        cv2.imwrite(filtered_image_path, bfilter)
        proses2 = filtered_image_path
        print(f'Edged image saved to: {filtered_image_path}')
        # ======================================================================



        # ======================================================================
        edged = cv2.Canny(bfilter, 30, 200)  # Edge detection
        edged_folder = os.path.join( 'static', 'edged_images')
        os.makedirs(edged_folder, exist_ok=True)
        edged_image_path = os.path.join(edged_folder, os.path.basename(filepath))
        cv2.imwrite(edged_image_path, edged)
        proses3 = edged_image_path
        print(f'Edged image saved to: {edged_image_path}')
        # ======================================================================

        keypoints = cv2.findContours(edged.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        contours = imutils.grab_contours(keypoints)
        contours = sorted(contours, key=cv2.contourArea, reverse=True)[:10]

        location = None
        for contour in contours:
            approx = cv2.approxPolyDP(contour, 10, True)
            if len(approx) == 4:
                location = approx
                break

        if location is None:
            raise ValueError("No valid contours found for OCR")

        mask = np.zeros(gray.shape, np.uint8)
        new_image = cv2.drawContours(mask, [location], 0, 255, -1)
        new_image = cv2.bitwise_and(img, img, mask=mask)

        (x, y) = np.where(mask == 255)
        (x1, y1) = (np.min(x), np.min(y))
        (x2, y2) = (np.max(x), np.max(y))

        cropped_image = gray[x1:x2+1, y1:y2+1]

        # Print cropped image dimensions for debugging
        print("Cropped image shape:", cropped_image.shape)

        # Create folder if it does not exist
        output_folder = os.path.join( 'static', 'cropped_images')
        os.makedirs(output_folder, exist_ok=True)

        # Save cropped image
        cropped_image_path = os.path.join(output_folder, os.path.basename(filepath))
        cv2.imwrite(cropped_image_path, cropped_image)
        print(f'Cropped image saved to: {cropped_image_path}')

        proses4 = cropped_image_path

        # Initialize EasyOCR reader
        reader = easyocr.Reader(['en'])

        # Perform OCR on cropped image
        result = reader.readtext(cropped_image)

        # Prepare the output dictionary
        output = {
            'proses1': proses1,
            'proses2': proses2,
            'proses3': proses3,
            'proses4': proses4,
            'ocr_result': result[0][-2] if result else "No text found"
        }

        return output

    except Exception as e:
        print("Error during image processing or OCR:", e)
        return {
            'proses1': None,
            'proses2': None,
            'proses3': None,
            'proses4': None,
            'ocr_result': "No text found"
        }



