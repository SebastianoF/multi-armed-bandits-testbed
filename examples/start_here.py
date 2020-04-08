import os
import shutil

from matplotlib import pyplot as plt

from mab.multi_armed_bandits import MultiArmedBandit, NonStationaryMultiArmedBandit


def stationary_benchmark(save_result=False):
    mab = MultiArmedBandit(seed=42)
    mab.get_plot_distributions()
    if save_result:
        plt.savefig("initial_distributions.pdf")
    else:
        plt.show()


def non_stationary_benchmark_slideshow(timepoints=10, output_folder="tmp_data"):

    if os.path.exists(output_folder):
        shutil.rmtree(output_folder, ignore_errors=True)
    os.mkdir(output_folder)

    # For this example we update the values for each timepoint, as there are no extractions.
    mab = NonStationaryMultiArmedBandit(seed=10, updates_intervals=range(timepoints))

    frames_list = []

    for t in range(timepoints):
        mab.get_plot_distributions(y_axis_limit=(-20, 20), timepoint=t)
        plt.ioff()
        pfi_frame = os.path.abspath(os.path.join(output_folder, 'step_{}.jpg'.format(t)))
        plt.savefig(pfi_frame)
        frames_list.append("file '" + pfi_frame + "'")
        if t != timepoints - 1:  # update the parameters each run but the last one.
            mab.update_parameters_random()

    pfi_frames_list = os.path.abspath(os.path.join(output_folder, 'frames_list.txt'))

    with open(pfi_frames_list, "w+") as outfile:
        outfile.write("\n".join(frames_list))

    pfi_output_gif = os.path.abspath(os.path.join(output_folder, 'sequence.gif'))
    os.system(f"ffmpeg -r 3 -f concat -safe 0 -i {pfi_frames_list} -y {pfi_output_gif}")
    print(f"gif created and stored in {pfi_output_gif}")

    print("historical means")
    print(mab.historical_means)

    print("historical stds")
    print(mab.historical_stds)


if __name__ == "__main__":
    # stationary_benchmark()
    non_stationary_benchmark_slideshow()
