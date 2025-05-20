import cv2
import os

def review_video_frames(video_path,
                        accepted_dir="data/negative_samples",
                        rejected_dir="data/rejected_frames",
                        second_skip=0.75):
    os.makedirs(accepted_dir, exist_ok=True)
    os.makedirs(rejected_dir, exist_ok=True)

    video_name = os.path.splitext(os.path.basename(video_path))[0]
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) * second_skip
    frame_index = 0
    accepted_count = 0
    rejected_count = 0
    rotation_angle = 0  # 0, 90, 180, 270

    while cap.isOpened():
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
        ret, frame = cap.read()
        if not ret:
            break

        # Apply rotation
        rotated = frame
        if rotation_angle == 90:
            rotated = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
        elif rotation_angle == 180:
            rotated = cv2.rotate(frame, cv2.ROTATE_180)
        elif rotation_angle == 270:
            rotated = cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)

        # Prepare display at half resolution
        h, w = rotated.shape[:2]
        display = cv2.resize(rotated, (w // 2, h // 2), interpolation=cv2.INTER_AREA)
        cv2.putText(display, f"Frame {frame_index}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.imshow("Frame", display)

        key = cv2.waitKey(0) & 0xFF

        if key == ord("a"):  # Accept
            fname = f"{video_name}_{accepted_count}.jpg"
            cv2.imwrite(os.path.join(accepted_dir, fname), rotated)
            accepted_count += 1
            frame_index += int(fps)

        elif key == ord("d"):  # Reject
            fname = f"{video_name}_{rejected_count}.jpg"
            cv2.imwrite(os.path.join(rejected_dir, fname), rotated)
            rejected_count += 1
            frame_index += int(fps)

        elif key == ord("r"):  # Rotate 90°
            rotation_angle = (rotation_angle + 90) % 360
            print(f"Rotated to {rotation_angle}°")

        elif ord("0") <= key <= ord("9"):  # Skip n seconds
            skip = key - ord("0")
            frame_index += int(skip * fps)

        elif key == 27:  # ESC
            break

        else:
            frame_index += int(fps)

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    root_path = "data/videos/negative_samples"
    for video_file in os.listdir(root_path):
        video_path = os.path.join(root_path, video_file)
        if os.path.isfile(video_path) and video_file.endswith(('.mp4', '.avi', '.mov')):
            print(f"Processing {video_file}...")
            review_video_frames(video_path)
            # move video to a different folder
            os.makedirs(os.path.join(root_path, "processed"), exist_ok=True)
            os.rename(video_path, os.path.join(root_path, "processed", video_file))
