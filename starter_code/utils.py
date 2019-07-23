from pathlib import Path

import nibabel as nib


def get_full_case_id(cid):
    try:
        cid = int(cid)
        case_id = "case_{:05d}".format(cid)
    except ValueError:
        case_id = cid

    return case_id


def get_case_path(source_path, cid):
    # Resolve location where data should be living
    data_path = Path(source_path)/ "data"
    if not data_path.exists():
        raise IOError(
            "Data path, {}, could not be resolved".format(str(data_path))
        )

    # Get case_id from provided cid
    case_id = get_full_case_id(cid)

    # Make sure that case_id exists under the data_path
    case_path = data_path / case_id
    if not case_path.exists():
        raise ValueError(
            "Case could not be found \"{}\"".format(case_path.name)
        )

    return case_path


def load_volume(source_path, cid):
    case_path = get_case_path(source_path, cid)
    vol = nib.load(str(case_path / "imaging.nii.gz"))
    return vol


def load_segmentation(source_path, cid):
    case_path = get_case_path(source_path, cid)
    seg = nib.load(str(case_path / "segmentation.nii.gz"))
    return seg


def load_case(cid,source_path):
    vol = load_volume(source_path, cid)
    seg = load_segmentation(source_path, cid)
    return vol, seg

