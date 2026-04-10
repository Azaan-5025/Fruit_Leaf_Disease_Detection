import os
import shutil

def consolidate():
    target_root = r"c:\Users\Syed Azaan Hussain\DS_Project\data\training_data"
    
    # 1. PlantVillage (Flat structure)
    pv_source = r"c:\Users\Syed Azaan Hussain\DS_Project\data\PlantVillage"
    pv_classes = [
        'Pepper__bell___Bacterial_spot', 'Pepper__bell___healthy', 'Potato___Early_blight',
        'Potato___Late_blight', 'Potato___healthy', 'Tomato_Bacterial_spot', 
        'Tomato_Early_blight', 'Tomato_Late_blight', 'Tomato_Leaf_Mold', 
        'Tomato_Septoria_leaf_spot', 'Tomato_Spider_mites_Two_spotted_spider_mite', 
        'Tomato__Target_Spot', 'Tomato__Tomato_YellowLeaf__Curl_Virus', 
        'Tomato__Tomato_mosaic_virus', 'Tomato_healthy'
    ]

    # 2. Fruits (Nested structure: Fruit/Condition)
    fruit_source = r"c:\Users\Syed Azaan Hussain\DS_Project\data\fruits\Dataset\train"
    fruit_classes = ['Apple', 'Banana', 'Grape', 'Mango', 'Orange']

    if not os.path.exists(target_root):
        os.makedirs(target_root)

    # Process PlantVillage
    for cls in pv_classes:
        src_path = os.path.join(pv_source, cls)
        if os.path.exists(src_path):
            dst_path = os.path.join(target_root, cls)
            print(f"Copying PV class: {cls}...")
            if os.path.exists(dst_path): shutil.rmtree(dst_path)
            shutil.copytree(src_path, dst_path)

    # Process Fruits (Flattening)
    for fruit in fruit_classes:
        fruit_path = os.path.join(fruit_source, fruit)
        if os.path.exists(fruit_path):
            # Iterate through subfolders like Fresh, Rotten, etc.
            for condition in os.listdir(fruit_path):
                cond_path = os.path.join(fruit_path, condition)
                if os.path.isdir(cond_path):
                    # Create a flattened class name: Banana___Fresh
                    new_cls_name = f"{fruit}___{condition}"
                    dst_path = os.path.join(target_root, new_cls_name)
                    print(f"Copying Fruit class: {new_cls_name}...")
                    if os.path.exists(dst_path): shutil.rmtree(dst_path)
                    shutil.copytree(cond_path, dst_path)

if __name__ == "__main__":
    consolidate()
