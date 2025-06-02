from __future__ import absolute_import
from setuptools import setup, find_packages
from setuptools.command.install import install


class InstallCommand(install):
    user_options = install.user_options + [
        ('no-ml', None, "Don't install without Machine Learning modules."),
    ]

    boolean_options = install.boolean_options + ['no-ml']

    def initialize_options(self):
        install.initialize_options(self)
        self.no_ml = None

    def finalize_options(self):
        install.finalize_options(self)
        if self.no_ml:
            dist = self.distribution
            dist.packages=find_packages(exclude=[
                "tests",
                "tests.*",
                "talon.signature",
                "talon.signature.*",
            ])
            for not_required in ["numpy", "scipy", "scikit-learn==0.24.1"]:
                dist.install_requires.remove(not_required)


setup(name='talon',
      version='1.6.0',
      description=("Mailgun library "
                   "to extract message quotations and signatures."),
      long_description=open("README.md").read(),
      author='Mailgun Inc.',
      author_email='admin@mailgunhq.com',
      url='https://github.com/mailgun/talon',
      license='APACHE2',
      cmdclass={
          'install': InstallCommand,
      },
      packages=find_packages(exclude=['tests', 'tests.*']),
      include_package_data=True,
      zip_safe=True,      install_requires=[
          "lxml>=2.3.3",
          "regex>=1",
          "numpy",
          "flask>=2.0.0",
          "markdownify>=0.11.6",
          "joblib",
          "scipy",
          "scikit-learn>=1.0.0",
          'charset-normalizer>=3.0.0',
          'cssselect',
          'six>=1.10.0',
          'html5lib'
          ],      tests_require=[
          "pytest>=7.0.0",
          "pytest-cov>=4.0.0",
          "coverage>=7.0.0"
          ]
      )
