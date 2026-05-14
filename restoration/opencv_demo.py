import cv2


def main() -> None:
    # Open default webcam (index 0). Change to 1/2 if needed.
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Kamera acilamadi. Farkli bir index deneyin (0/1/2).")
        return

    print("Cikis icin 'q' tusuna basin.")
    while True:
        ok, frame = cap.read()
        if not ok:
            print("Kare okunamadi, cikiliyor.")
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 80, 160)

        cv2.imshow("Orijinal", frame)
        cv2.imshow("Kenarlar (Canny)", edges)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
