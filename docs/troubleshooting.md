# Troubleshooting

## C++ Compiler

The pytorch inline CUDA compiler sometimes uses `c++` executable, not `g++`. Make sure to do the following to allow pytorch to compile the inline CUDA code:

```bash
mkdir -p $HOME/bin
ln -sf "$(which g++)" $HOME/bin/c++

# add to .bashrc
export PATH=$HOME/bin:$PATH
```