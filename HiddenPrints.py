import os, sys
# https://stackoverflow.com/questions/8391411/how-to-block-calls-to-print#:~:text=If%20you%20don't%20want,the%20top%20of%20the%20file.

class HiddenPrints:
    def __enter__(self):
        self._original_stdout = sys.stdout
        sys.stdout = open(os.devnull, 'w')

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout.close()
        sys.stdout = self._original_stdout