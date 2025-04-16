from collections.abc import Generator
from typing import Any
import os
from dify_plugin import Tool
from dify_plugin.entities.tool import ToolInvokeMessage
from utils.vanna import RookieVanna
from openai import OpenAI


os.environ["TOKENIZERS_PARALLELISM"] = "false"  # 修复并行性警告


class RookieVannaTool(Tool):
    def _invoke(self, tool_parameters: dict[str, Any]) -> Generator[ToolInvokeMessage]:
        client = OpenAI(
            api_key=tool_parameters['api_key'],
            base_url=tool_parameters['base_url']
        )
        vn = RookieVanna(client=client,config={'model':tool_parameters['model'],'temperature':0.7})
        vn.connect_to_mysql(host=tool_parameters['host'], dbname=tool_parameters['db_name'], user=tool_parameters['username'], password=tool_parameters['password'], port=tool_parameters['port'])
        training_data = vn.get_training_data()
        similar_question_sql = vn.get_similar_question_sql(tool_parameters['query'])
        if similar_question_sql is not None:
            sql_valid = vn.is_sql_valid(similar_question_sql[0]['sql'])
            if sql_valid:
                yield self.create_json_message({
                    "result": similar_question_sql[0]['sql']
                })
                return
            else:
                raise ValueError(f"SQL 生成失败：{str(similar_question_sql[0]['sql'])}")
                
        generate_sql = vn.generate_sql(tool_parameters['query'])
        yield self.create_json_message({
            "result": generate_sql
        })
