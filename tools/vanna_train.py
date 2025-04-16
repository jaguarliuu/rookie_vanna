from collections.abc import Generator
from typing import Any

from dify_plugin import Tool
from dify_plugin.entities.tool import ToolInvokeMessage
from utils.vanna import RookieVanna
from openai import OpenAI

class VannaTrainTool(Tool):
    def _invoke(self, tool_parameters: dict[str, Any]) -> Generator[ToolInvokeMessage]:
        client = OpenAI(
            api_key=tool_parameters['api_key'],
            base_url=tool_parameters['base_url']
        )
        vn = RookieVanna(client=client,config={'model':tool_parameters['model'],'temperature':0.7})
        vn.connect_to_mysql(host=tool_parameters['host'], dbname=tool_parameters['db_name'], user=tool_parameters['username'], password=tool_parameters['password'], port=tool_parameters['port'])
        if tool_parameters['train_type']=='global':
            # The information schema query may need some tweaking depending on your database. This is a good starting point.
            df_information_schema = vn.run_sql("SELECT * FROM INFORMATION_SCHEMA.COLUMNS")
            # This will break up the information schema into bite-sized chunks that can be referenced by the LLM
            plan = vn.get_training_plan_generic(df_information_schema)
            vn.train(plan=plan)
        if tool_parameters['train_type']=='qa':
            vn.train(
                question=tool_parameters['train_question'],
                sql=tool_parameters['train_sql']
            )
        
        yield self.create_json_message({
            "result": "Train Success!"
        })
