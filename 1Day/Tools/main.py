# from langchain_community.tools import DuckDuckGoSearchRun

# # Corrected the variable name spelling to search_tool
# search_tool = DuckDuckGoSearchRun()

# # Invoke the tool to get the results
# results = search_tool.invoke("wwb")
# print(results)

#---------------------------------------------------------------------------------


from langchain_community.tools import ShellTool # help run command on the computer in code is running

# Corrected the variable name spelling to search_tool
shell_tool = ShellTool()

# Invoke the tool to get the results
results = shell_tool.invoke("whoami")
print(results)
