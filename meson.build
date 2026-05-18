project('gaussregular', 'c',
  version : '0.1.3',
  default_options : ['warning_level=2', 'buildtype=release', 'b_lto=true'])

py_mod = import('python')
py = py_mod.find_installation('python', pure: false)
py_dep = py.dependency()

incdir_numpy_cmd = run_command(
  py,
  ['-c', 'import numpy; print(numpy.get_include())'],
  check: true,
)
incdir_numpy = incdir_numpy_cmd.stdout().strip()
inc_np = include_directories(incdir_numpy)

host_system = host_machine.system()
cc = meson.get_compiler('c')
fast_math = get_option('fast_math')

c_args = [
  '-DNPY_NO_DEPRECATED_API=NPY_1_19_API_VERSION',
]

if host_system == 'windows' and cc.get_id() == 'msvc'
  c_args += ['/O2', '/GL', '/Oi', '/Ot']
  if fast_math
    c_args += ['/fp:fast']
  else
    c_args += ['/fp:precise']
  endif
else
  c_args += ['-O3', '-fno-math-errno']
  if fast_math
    c_args += ['-ffast-math']
  endif
endif

core_ext = py.extension_module(
  '_core',
  'src/gaussregular/_core.c',
  include_directories: inc_np,
  dependencies: [py_dep],
  c_args: c_args,
  install: true,
  subdir: 'gaussregular',
)

py.install_sources(
  ['src/gaussregular/__init__.py', 'src/gaussregular/api.py'],
  subdir: 'gaussregular',
)

py.install_sources(
  ['src/gaussregular/__init__.py'],
  subdir: 'gaussreg',
)
