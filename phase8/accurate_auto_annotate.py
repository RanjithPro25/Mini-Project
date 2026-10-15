try:

    start_time = time.time()

    for index, task in enumerate(tasks, 1):

        split = task["split"]
        source = task["source"]
        species = task["species"]

        image_name = source.name

        image_file = (
            IMAGES_DIR /
            split /
            image_name
        )

        label_file = (
            LABELS_DIR /
            split /
            f"{source.stem}.txt"
        )

        key = f"{split}|{task['destination']}"

        if valid_label(label_file):

            preserved += 1

            manifest[key] = {
                "split": split,
                "destination": str(task["destination"]),
                "species": species,
                "label": str(label_file.relative_to(YOLO_DIR)),
                "status": "existing",
                "confidence_1": "existing",
                "confidence_2": "existing",
                "confidence_3": "existing",
                "box_iou": "existing",
                "attempt": "0",
                "source": str(source)
            }

            print(
                f"[{index:,}/{total:,}] "
                f"KEEP | {split} | {species} | {image_name}",
                flush=True
            )

            continue

        print(
            f"[{index:,}/{total:,}] "
            f"PROCESSING | {split} | {species} | {image_name}",
            flush=True
        )

        image_start = time.time()

        try:

            image = cv2.imread(str(image_file))

            if image is None:
                raise RuntimeError("Unable to read image")

            height, width = image.shape[:2]

            print(
                "    Attempt 1: 640px...",
                flush=True
            )

            d1 = detect(
                image_file,
                CONF_1,
                IMGSZ_1
            )

            if d1 is not None:

                print(
                    f"    Detection 1: "
                    f"{d1['confidence']:.3f}",
                    flush=True
                )

            else:

                print(
                    "    Detection 1: NONE",
                    flush=True
                )

            print(
                "    Attempt 2: 960px...",
                flush=True
            )

            d2 = detect(
                image_file,
                CONF_2,
                IMGSZ_2
            )

            if d2 is not None:

                print(
                    f"    Detection 2: "
                    f"{d2['confidence']:.3f}",
                    flush=True
                )

            else:

                print(
                    "    Detection 2: NONE",
                    flush=True
                )

            if d1 is not None and d2 is not None:

                b1 = d1["box"]
                b2 = d2["box"]

                c1 = d1["confidence"]
                c2 = d2["confidence"]

                overlap = calculate_iou(
                    b1,
                    b2
                )

                print(
                    f"    Box IoU: {overlap:.3f}",
                    flush=True
                )

                if (
                    valid_box(
                        b1,
                        width,
                        height
                    )
                    and
                    valid_box(
                        b2,
                        width,
                        height
                    )
                    and
                    overlap >= BOX_IOU_MIN
                    and
                    c1 >= CONF_MIN_1
                    and
                    c2 >= CONF_MIN_2
                ):

                    final_box = (
                        b1 + b2
                    ) / 2.0

                    save_label(
                        label_file,
                        final_box,
                        width,
                        height
                    )

                    manifest[key] = {
                        "split": split,
                        "destination": str(task["destination"]),
                        "species": species,
                        "label": str(
                            label_file.relative_to(
                                YOLO_DIR
                            )
                        ),
                        "status": "accepted",
                        "confidence_1": f"{c1:.6f}",
                        "confidence_2": f"{c2:.6f}",
                        "confidence_3": "",
                        "box_iou": f"{overlap:.6f}",
                        "attempt": "2",
                        "source": str(source)
                    }

                    accepted += 1

                    elapsed = time.time() - image_start

                    print(
                        f"    RESULT: ACCEPTED | "
                        f"{elapsed:.1f}s",
                        flush=True
                    )

                    save_manifest()
                    save_progress()

                    continue

            print(
                "    Attempt 3: 1280px recovery...",
                flush=True
            )

            d3 = detect(
                image_file,
                CONF_3,
                IMGSZ_3
            )

            if d3 is not None:

                b3 = d3["box"]
                c3 = d3["confidence"]

                print(
                    f"    Recovery confidence: "
                    f"{c3:.3f}",
                    flush=True
                )

                if (
                    valid_box(
                        b3,
                        width,
                        height
                    )
                    and
                    c3 >= CONF_MIN_3
                ):

                    save_label(
                        label_file,
                        b3,
                        width,
                        height
                    )

                    manifest[key] = {
                        "split": split,
                        "destination": str(task["destination"]),
                        "species": species,
                        "label": str(
                            label_file.relative_to(
                                YOLO_DIR
                            )
                        ),
                        "status": "accepted_recovery",
                        "confidence_1": (
                            f"{d1['confidence']:.6f}"
                            if d1 else ""
                        ),
                        "confidence_2": (
                            f"{d2['confidence']:.6f}"
                            if d2 else ""
                        ),
                        "confidence_3": f"{c3:.6f}",
                        "box_iou": "",
                        "attempt": "3",
                        "source": str(source)
                    }

                    recovery += 1

                    print(
                        "    RESULT: "
                        "ACCEPTED AFTER RECOVERY",
                        flush=True
                    )

                else:

                    manifest[key] = {
                        "split": split,
                        "destination": str(task["destination"]),
                        "species": species,
                        "label": "",
                        "status": "review",
                        "confidence_1": (
                            f"{d1['confidence']:.6f}"
                            if d1 else ""
                        ),
                        "confidence_2": (
                            f"{d2['confidence']:.6f}"
                            if d2 else ""
                        ),
                        "confidence_3": f"{c3:.6f}",
                        "box_iou": "",
                        "attempt": "3",
                        "source": str(source)
                    }

                    review += 1

                    print(
                        "    RESULT: REVIEW",
                        flush=True
                    )

            else:

                manifest[key] = {
                    "split": split,
                    "destination": str(task["destination"]),
                    "species": species,
                    "label": "",
                    "status": "no_detection",
                    "confidence_1": (
                        f"{d1['confidence']:.6f}"
                        if d1 else "0"
                    ),
                    "confidence_2": (
                        f"{d2['confidence']:.6f}"
                        if d2 else "0"
                    ),
                    "confidence_3": "0",
                    "box_iou": "",
                    "attempt": "3",
                    "source": str(source)
                }

                no_detection += 1

                print(
                    "    RESULT: NO DETECTION",
                    flush=True
                )

            elapsed = time.time() - image_start

            print(
                f"    Time: {elapsed:.1f}s",
                flush=True
            )

        except Exception as e:

            manifest[key] = {
                "split": split,
                "destination": str(task["destination"]),
                "species": species,
                "label": "",
                "status": "failed",
                "confidence_1": "error",
                "confidence_2": "error",
                "confidence_3": "error",
                "box_iou": "",
                "attempt": "error",
                "source": str(source)
            }

            failed += 1

            print(
                f"    RESULT: FAILED",
                flush=True
            )

            print(
                f"    ERROR: {e}",
                flush=True
            )

        save_manifest()
        save_progress()

        elapsed_total = time.time() - start_time

        completed = (
            preserved +
            accepted +
            recovery +
            review +
            no_detection +
            failed
        )

        if completed > 0:

            avg = elapsed_total / completed

            remaining = total - completed

            eta_seconds = avg * remaining

            eta_minutes = eta_seconds / 60

            print(
                f"    PROGRESS: "
                f"{completed:,}/{total:,} "
                f"({completed / total * 100:.2f}%) | "
                f"ETA: {eta_minutes:.1f} min",
                flush=True
            )

            print(
                "-" * 75,
                flush=True
            )

except KeyboardInterrupt:

    save_manifest()
    save_progress()

    print()
    print("=" * 75)
    print("INTERRUPTED SAFELY")
    print("=" * 75)
    print("Progress saved.")
    print("Run the same command again to continue.")
    print()

    raise SystemExit(0)