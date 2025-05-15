# Replace conda/mamba with py-rattler for package management

This PR replaces conda/mamba with py-rattler for package management and environment handling in snakedeploy.

## Changes

- Added py-rattler as a dependency in pyproject.toml
- Added imports for py-rattler modules in conda.py
- Modified the CondaEnvProcessor class to exclusively use py-rattler
- Removed the conda-frontend CLI option
- Updated function signatures to maintain backward compatibility

## Implementation Details

The implementation completely replaces conda/mamba subprocess calls with py-rattler API calls, which provides better performance and error handling. The py-rattler API is used to:

- Create and remove conda environments
- List packages in environments
- Pin environments with exact package URLs

## Testing

The implementation has been tested manually to ensure it works correctly with the existing codebase.

## Link to Devin run
https://app.devin.ai/sessions/e232a379875343b79ab4ffdcb550f384

Requested by: Jake VanCampen
