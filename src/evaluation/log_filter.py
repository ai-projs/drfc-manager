import sys

class LogFilter:
    """Filters out noisy Docker logs to make evaluation output cleaner."""
    
    @staticmethod
    def filter_docker_output(output: str) -> str:
        """Filter common Docker warning messages from output."""
        if not output:
            return output
            
        filtered_lines = []
        for line in output.split('\n'):
            # Skip docker compose variable not set warnings
            if ('level=warning msg="The ' in line and 'variable is not set' in line) or \
               ('level=warning msg="Warning: No resource found' in line) or \
               ('version` is obsolete' in line):
                continue
            filtered_lines.append(line)
        
        return '\n'.join(filtered_lines)