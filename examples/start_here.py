import os
import shutil

from matplotlib import pyplot as plt

from mab.multi_armed_bandits import MultiArmedBandit


def see_a_static_mab():
    mab = MultiArmedBandit(seed=10)
    mab.get_plt_distribution()
    plt.show()


def get_a_slideshow_of_changing_mab(timepoints=10, output_folder="tmp_data"):

    if os.path.exists(output_folder):
        shutil.rmtree(output_folder, ignore_errors=True)
    os.mkdir(output_folder)

    mab = MultiArmedBandit(seed=10)

    frames_list = []

    for t in range(timepoints):
        mab.get_plt_distribution(y_axis_limit=(-20, 20), timepoint=t)
        plt.ioff()
        pfi_frame = os.path.abspath(os.path.join(output_folder, 'step_{}.jpg'.format(t)))
        plt.savefig(pfi_frame)
        frames_list.append("file '" + pfi_frame + "'")
        mab.update_parameters_random()

    pfi_frames_list = os.path.abspath(os.path.join(output_folder, 'frames_list.txt'))

    with open(pfi_frames_list, "w+") as outfile:
        outfile.write("\n".join(frames_list))

    pfi_output_gif = os.path.abspath(os.path.join(output_folder, 'sequence.gif'))
    os.system(f"ffmpeg -r 3 -f concat -safe 0 -i {pfi_frames_list} -y {pfi_output_gif}")
    print(f"gif created and stored in {pfi_output_gif}")


if __name__ == "__main__":
    see_a_static_mab()
    get_a_slideshow_of_changing_mab()
