from pathlib import Path
import yaml
from src.image_utils import list_images
from src.patchcore_model import SmartPatchCore


def main():
    # تحديد مسار المشروع الرئيسي
    project_dir = Path(__file__).resolve().parent

    # قراءة ملف التكوين
    config_path = project_dir / "config.yaml"
    if not config_path.exists():
        print(f"ERROR: config.yaml not found at {config_path}")
        return

    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    training_config = config["training"]
    model_config = config["model"]

    # مسار النموذج المدرب
    model_dir = Path(training_config.get("model_dir", "models/trained_model"))
    if not model_dir.is_absolute():
        model_dir = project_dir / model_dir

    if not model_dir.exists():
        print(f"ERROR: Trained model not found at {model_dir}. Please run train.py first.")
        return

    print("=" * 60)
    print("          SMARTINSPECT AI - INFERENCE & TESTING")
    print("=" * 60)
    print(f"Loading model from: {model_dir}")

    # تحميل النموذج المدرب مسبقاً
    model = SmartPatchCore(
        image_size=model_config["image_size"],
        coreset_ratio=model_config["coreset_ratio"],
        neighbors=model_config["neighbors"],
        threshold_percentile=model_config["threshold_percentile"],
    )
    model.load(str(model_dir))
    print("Model loaded successfully!")

    # تحديد مجلد الصور المراد اختبارها (مثلاً صور الاختبار أو العيوب)
    # يمكنك إنشاء مجلد باسم data/test/mixed أو وضع الصور هناك
    test_dir = project_dir / "data" / "test"
    
    if not test_dir.exists():
        print(f"\nWARNING: Test directory not found at {test_dir}.")
        print("Please create 'data/test/' and put images inside to test them.")
        return

    test_image_paths = list_images(str(test_dir))
    if not test_image_paths:
        print(f"No images found for testing in {test_dir}")
        return

    print(f"Found {len(test_image_paths)} images to test. Running inference...\n")

    # اختبار كل صورة ومعرفة ما إذا كانت سليمة أم معيبة
    for img_path in test_image_paths:
        is_anomaly, score = model.predict(img_path)
        status = "DEFECTIVE (معيب ⚠️)" if is_anomaly else "NORMAL (سليم ✅)"
        print(f"Image: {Path(img_path).name} --> Status: {status} | Score: {score:.4f}")

    print("\n" + "=" * 60)
    print("INFERENCE COMPLETED!")
    print("=" * 60)


if __name__ == "__main__":
    main()