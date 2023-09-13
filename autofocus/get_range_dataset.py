import numpy as np
import supervisely as sly

if __name__ == "__main__":
    project_path = r"C:\Users\tristan_cotte\PycharmProjects\prior_controller\autofocus\sly_project"
    project = sly.Project(project_path, sly.OpenMode.READ)
    meta = project.meta

    for ds in project.datasets:
        print(ds.name)
        min = np.inf
        max = -np.inf

        for item in ds.items():
            item_name, picture_path, json_path = item
            annotation = sly.Annotation.load_json_file(json_path, meta)
            z_value = annotation.img_tags.get('focus_difference').value

            if z_value > max:
                max = z_value
            if z_value < min:
                min = z_value


        print(f"Dataset {ds.name} -- Z range [{min} ; {max}]")


    """
    ds0
    Dataset ds0 -- Z range [-145 ; 440]
    ds1
    Dataset ds1 -- Z range [-360 ; 350]
    ds2
    Dataset ds2 -- Z range [-170 ; 370]
    min = -145
    max = 350
    """
