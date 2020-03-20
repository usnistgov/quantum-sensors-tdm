from distutils.core import setup, Extension #@UnusedImport

setup(name="nasa_client",
      packages=['nasa_client'],
      package_dir={'nasa_client':'.'},
      py_modules=['easyClient','easyClientNDFB','easyClientDastard',
      'extern_trig_client','rpc_client_for_easy_client']
      )
