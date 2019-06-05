#array containing regular expressions for common scanning errors and the corrected strings
#TODO add RE for <end of sentence     .Begin of next sentence> you'd and other would
fixes = [("\s*'\s*t\s*", "'t "), 
         ("\s*'\s*ve\s*", "'ve "), 
         ("\s*'\s*s\s*", "'s "), 
         ("s\s*'\s*", "s' "), 
         ("\s*I\s*'\s*m\s*", "I'm "), 
         ("\s*'\s*nt\s*", "n't "), 
         ("\s*\.{3}s*", "... "), 
         ("\s*\?\s*", "? "),
         ("\s*!\s*", "! "), 
         ("\s*,\s*", ", "), 
         ("\s*\.\s*", ". "), 
         ("\s*;\s*", "; "), 
         ("\s*:\s*", ": "), 
         ("s\s*’\s*", "s' "),
         ("\s*\(\s*", "("),
         ("\s*\)\s*", ")"),
         ("you\s*\'\s*d",'you\'d')]

