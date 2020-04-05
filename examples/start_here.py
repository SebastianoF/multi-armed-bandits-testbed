import os

from mab.multi_armed_bandits import MultiArmedBandit


def see_a_static_mab():
    mab = MultiArmedBandit(seed=10)
    p = mab.get_plt_distribution()
    p.show()


def get_a_slideshow_of_changing_mab(timepoints=50, output_folder="tmp_data"):
    mab = MultiArmedBandit(seed=10)

    for t in range(timepoints):
        p = mab.get_plt_distribution()
        mab.update_parameters_random()

        p.savefig(
            os.path.join(output_folder, 'step_{}.jpg'.format(t)
        )


if __name__ == "__main__":

    see_a_static_mab()

